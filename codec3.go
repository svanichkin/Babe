package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/klauspost/compress/zstd"
)

const (
	magicCodec2 = "BAB2"
)

func paramsForQuality(q int) codec2Params {
	if q < 1 {
		q = 1
	}
	if q > 100 {
		q = 100
	}

	qi := 100 - q

	params := codec2Params{
		minBlock:           1 + int(float64(qi)*(4.0/100.0)),
		maxGrad:            int32(ExpMap(qi, 99, 0, 150, 1, -4.0)),
		colorTol:           ExpMap(qi, 99, 0, 7, 1, -3.0),
		patternGainPercent: ExpMap(q, 1, 100, 0, 100, 3.0), // по умолчанию: паттерн должен быть лучше на 50%
	}
	fmt.Printf("QUALITY: minBlock=%d, maxGrad=%d, colorTol=%d, patternFactor=%d\n",
		params.minBlock,
		params.maxGrad,
		params.colorTol,
		params.patternGainPercent,
	)
	return params
}

// Encode encodes the given image with the given quality (1–100).
// Higher quality => smaller allowed luma spread => больше блоков, но лучше детализация.
func Encode(img image.Image, quality int) ([]byte, error) {
	quality = ValidateQuality(quality)
	params := paramsForQuality(quality)

	// Конвертируем исходник один раз в full-res YCbCr 4:4:4 и дальше работаем в этом пространстве.
	yimg := ImageToYCbCr(img)
	luma := buildLumaBuffer(yimg)

	w := uint16(yimg.Bounds().Dx())
	h := uint16(yimg.Bounds().Dy())

	b := &bytes.Buffer{}

	stripesHint := runtime.NumCPU()
	if stripesHint < 1 {
		stripesHint = 1
	}
	if stripesHint > 255 {
		stripesHint = 255
	}

	// Write header
	if err := WriteHeader(b, quality, w, h, uint8(stripesHint)); err != nil {
		return nil, err
	}

	// 1) Первый проход: параллельно собираем цвета для всех листьев и форму дерева по вертикальным полосам.
	leaves, pattLeaves, leafModes, pattern, stats, err := collectRegionColorsStriped(yimg, luma, params, stripesHint)
	if err != nil {
		return nil, err
	}

	if stats.total > 0 {
		frac := float64(stats.patternBetter) * 100.0 / float64(stats.total)
		fmt.Printf("leaf stats: total=%d, pattern-better=%d (%.1f%%)\n",
			stats.total, stats.patternBetter, frac)
	}

	// 2) Строим набор локальных палитр и локальные индексы листьев.
	// Для будущей поддержки паттерн-листьев buildPalettes уже получает и pattLeaves, и leafModes,
	// но на этом шаге мы всё ещё используем только solid-индексы (idx).
	pals, refs, err := buildPalettes(leaves, pattLeaves, leafModes, params)
	if err != nil {
		return nil, err
	}

	// Считаем, сколько листьев пришлось на каждую палитру (для декодера).
	leafCounts := make([]uint32, len(pals))
	for _, r := range refs {
		pid := int(r.pal)
		if pid < 0 || pid >= len(pals) {
			return nil, fmt.Errorf("encode: invalid palette id %d in refs", pid)
		}
		leafCounts[pid]++
	}

	// 3) Собираем сырой битстрим.
	var raw bytes.Buffer

	// Сначала пишем все палитры:
	// uint16 numPalettes, затем для каждой:
	// uint16 leafCount, uint16 palSize, затем palSize * RGB.
	if err := binary.Write(&raw, binary.BigEndian, uint16(len(pals))); err != nil {
		return nil, err
	}
	for i, p := range pals {
		// Базовый инвариант: одна палитра содержит не более 256 уникальных цветов.
		if len(p.colors) > 256 {
			return nil, fmt.Errorf("_encode: palette size %d exceeds 256", len(p.colors))
		}
		// количество листьев, использующих эту палитру
		if err := binary.Write(&raw, binary.BigEndian, leafCounts[i]); err != nil {
			return nil, err
		}
		// размер палитры (uint16, чтобы вместить 256)
		if err := binary.Write(&raw, binary.BigEndian, uint16(len(p.colors))); err != nil {
			return nil, err
		}
		// Дельта-кодирование палитры по ПРЕДЫДУЩЕМУ цвету для уже отсортированных массивов
		if len(p.colors) > 0 {
			y0, co0, cg0 := packYCoCg(p.colors[0])
			if err := raw.WriteByte(y0); err != nil {
				return nil, err
			}
			if err := raw.WriteByte(co0); err != nil {
				return nil, err
			}
			if err := raw.WriteByte(cg0); err != nil {
				return nil, err
			}

			prevY := int(y0)
			prevCo := int(co0)
			prevCg := int(cg0)

			for j := 1; j < len(p.colors); j++ {
				y, co, cg := packYCoCg(p.colors[j])

				dy := int(y) - prevY
				dco := int(co) - prevCo
				dcg := int(cg) - prevCg

				if dy < -128 {
					dy = -128
				} else if dy > 127 {
					dy = 127
				}
				if dco < -128 {
					dco = -128
				} else if dco > 127 {
					dco = 127
				}
				if dcg < -128 {
					dcg = -128
				} else if dcg > 127 {
					dcg = 127
				}

				if err := raw.WriteByte(byte(dy + 128)); err != nil {
					return nil, err
				}
				if err := raw.WriteByte(byte(dco + 128)); err != nil {
					return nil, err
				}
				if err := raw.WriteByte(byte(dcg + 128)); err != nil {
					return nil, err
				}

				prevY = int(y)
				prevCo = int(co)
				prevCg = int(cg)
			}
		}
	}

	// 4) Второй проход: кодируем дерево, индексы и паттерны по тем же полосам, что и в collectRegionColorsStriped.
	// Структуру дерева пишем в отдельный буфер, индексы листьев — в другой, паттерн-биты — в третий.
	var treeBuf bytes.Buffer
	var leafBuf bytes.Buffer
	var patBuf bytes.Buffer

	bwTree := NewBitWriter(&treeBuf)
	bwPat := NewBitWriter(&patBuf)

	leafPos := 0
	patPos := 0

	stripes := splitStripes(int(w), int(h), params.minBlock, stripesHint)
	for _, s := range stripes {
		if err := encodeRegion(
			yimg, luma, s.x, s.y, s.w, s.h,
			params, bwTree, refs, &leafPos, pattern, &patPos, &leafBuf, bwPat,
		); err != nil {
			return nil, err
		}
	}
	if err := bwTree.Flush(); err != nil {
		return nil, err
	}
	if err := bwPat.Flush(); err != nil {
		return nil, err
	}

	// После палитр собираем итоговый поток:
	// [ uint32 treeLen ][ treeBits ][ uint32 leafLen ][ leafIndices ][ uint32 patLen ][ patBits ].
	treeBytes := treeBuf.Bytes()
	rawLeafBytes := leafBuf.Bytes()
	patBytes := patBuf.Bytes()

	// Delta-encode leaf index stream losslessly (mod 256).
	// Первый байт пишем как есть, далее храним разность по модулю 256.
	leafBytes := make([]byte, len(rawLeafBytes))
	if len(rawLeafBytes) > 0 {
		leafBytes[0] = rawLeafBytes[0]
		prev := int(rawLeafBytes[0])
		for i := 1; i < len(rawLeafBytes); i++ {
			cur := int(rawLeafBytes[i])
			// delta = (cur - prev) mod 256
			delta := (cur - prev) & 0xFF
			leafBytes[i] = byte(delta)
			prev = cur
		}
	}

	// Debug: размеры секций (до zstd)
	var palettesBytes int
	for _, p := range pals {
		palettesBytes += 2 // leafCount
		palettesBytes += 2 // palSize
		palettesBytes += len(p.colors) * 3
	}
	palettesBytes += 2 // numPalettes

	fmt.Printf("STREAM SIZES (raw, bytes): palettes=%d, leafIndices=%d, patternBits=%d, treeBits=%d\n",
		palettesBytes,
		len(leafBytes),
		len(patBytes),
		len(treeBytes),
	)

	if err := binary.Write(&raw, binary.BigEndian, uint32(len(treeBytes))); err != nil {
		return nil, err
	}
	if _, err := raw.Write(treeBytes); err != nil {
		return nil, err
	}

	if err := binary.Write(&raw, binary.BigEndian, uint32(len(leafBytes))); err != nil {
		return nil, err
	}
	if _, err := raw.Write(leafBytes); err != nil {
		return nil, err
	}

	if err := binary.Write(&raw, binary.BigEndian, uint32(len(patBytes))); err != nil {
		return nil, err
	}
	if _, err := raw.Write(patBytes); err != nil {
		return nil, err
	}

	// 5) Zstd
	if err := EncodeZstd(b, &raw); err != nil {
		return nil, err
	}

	return b.Bytes(), nil
}

// Decode decodes data produced by Encode.
func Decode(data []byte) (image.Image, error) {
	r := bytes.NewReader(data)

	quality, w, h, stripesHint, err := ReadHeader(r)
	if err != nil {
		return nil, err
	}

	quality = ValidateQuality(quality)
	params := paramsForQuality(quality)

	if stripesHint < 1 {
		stripesHint = 1
	}
	if stripesHint > 255 {
		stripesHint = 255
	}

	dst := image.NewRGBA(image.Rect(0, 0, w, h))

	// Оставшиеся данные — zstd-кадр с битстримом.
	plain, err := DecodeZstd(r)
	if err != nil {
		return nil, err
	}

	// читаем набор палитр
	reader := bytes.NewReader(plain)
	var numPalettes uint16
	if err := binary.Read(reader, binary.BigEndian, &numPalettes); err != nil {
		return nil, err
	}
	palettes := make([][]color.RGBA, numPalettes)
	leafCounts := make([]uint32, numPalettes)
	for i := 0; i < int(numPalettes); i++ {
		if err := binary.Read(reader, binary.BigEndian, &leafCounts[i]); err != nil {
			return nil, err
		}
		var sz16 uint16
		if err := binary.Read(reader, binary.BigEndian, &sz16); err != nil {
			return nil, err
		}
		sz := int(sz16)
		if sz < 0 {
			return nil, fmt.Errorf("codec2: negative palette size")
		}
		p := make([]color.RGBA, sz)
		// Дельта-декодирование палитры по ПРЕДЫДУЩЕМУ цвету
		if sz > 0 {
			var ycc [3]byte
			if _, err := io.ReadFull(reader, ycc[:]); err != nil {
				return nil, err
			}
			p[0] = unpackYCoCg(ycc[0], ycc[1], ycc[2])

			prevY := int(ycc[0])
			prevCo := int(ycc[1])
			prevCg := int(ycc[2])

			for j := 1; j < sz; j++ {
				var d [3]byte
				if _, err := io.ReadFull(reader, d[:]); err != nil {
					return nil, err
				}

				dy := int(int8(d[0] - 128))
				dco := int(int8(d[1] - 128))
				dcg := int(int8(d[2] - 128))

				y := prevY + dy
				co := prevCo + dco
				cg := prevCg + dcg

				if y < 0 {
					y = 0
				} else if y > 255 {
					y = 255
				}
				if co < 0 {
					co = 0
				} else if co > 255 {
					co = 255
				}
				if cg < 0 {
					cg = 0
				} else if cg > 255 {
					cg = 255
				}

				p[j] = unpackYCoCg(byte(y), byte(co), byte(cg))

				prevY = y
				prevCo = co
				prevCg = cg
			}
		}
		palettes[i] = p
	}

	// Далее в потоке:
	// uint32 treeLen, [treeLen] treeBytes,
	// uint32 leafLen, [leafLen] leafBytes,
	// uint32 patLen,  [patLen] patBytes.
	var treeLen uint32
	if err := binary.Read(reader, binary.BigEndian, &treeLen); err != nil {
		return nil, err
	}
	if int(treeLen) > reader.Len() {
		return nil, fmt.Errorf("codec2: invalid tree length")
	}
	treeBytes := make([]byte, treeLen)
	if _, err := io.ReadFull(reader, treeBytes); err != nil {
		return nil, err
	}

	var leafLen uint32
	if err := binary.Read(reader, binary.BigEndian, &leafLen); err != nil {
		return nil, err
	}
	if int(leafLen) > reader.Len() {
		return nil, fmt.Errorf("codec2: invalid leaf length")
	}
	encLeafBytes := make([]byte, leafLen)
	if _, err := io.ReadFull(reader, encLeafBytes); err != nil {
		return nil, err
	}

	// Delta-decode leaf index stream back to original indices (mod 256).
	leafBytes := make([]byte, leafLen)
	if leafLen > 0 {
		leafBytes[0] = encLeafBytes[0]
		prev := int(leafBytes[0])
		for i := 1; i < int(leafLen); i++ {
			delta := int(encLeafBytes[i]) // 0..255
			// cur = (prev + delta) mod 256
			cur := (prev + delta) & 0xFF
			leafBytes[i] = byte(cur)
			prev = cur
		}
	}

	var patLen uint32
	if err := binary.Read(reader, binary.BigEndian, &patLen); err != nil {
		return nil, err
	}
	if int(patLen) > reader.Len() {
		return nil, fmt.Errorf("codec2: invalid pattern length")
	}
	patBytes := make([]byte, patLen)
	if _, err := io.ReadFull(reader, patBytes); err != nil {
		return nil, err
	}

	treeBr := NewBitReaderFromBytes(treeBytes)
	patBr := NewBitReaderFromBytes(patBytes)
	leafPos := 0

	// Первый этап: последовательно читаем дерево по тем же вертикальным полосам
	// и собираем список листьев.
	var jobs []leafJob
	stripes := splitStripes(w, h, params.minBlock, stripesHint)
	curPal := 0
	usedInPal := 0
	for _, s := range stripes {
		if err := decodeRegionJobs(
			s.x, s.y, s.w, s.h,
			params,
			treeBr, patBr,
			dst,
			palettes,
			leafCounts,
			&curPal, &usedInPal,
			leafBytes, &leafPos,
			&jobs,
		); err != nil {
			return nil, err
		}
	}

	// Второй этап: параллельно заливаем блоки цветом.
	paintLeafJobsParallel(dst, palettes, jobs)

	// Постпроцессинг блоков: мягкое сглаживание границ и стыков.
	// Привязываем шаг сетки сглаживания к params.minBlock,
	// чтобы не лезть внутрь блоков мельче минимального.
	smoothed := smoothBlocks(dst, params.minBlock)
	return smoothed, nil
}

// -----------------------------------------------------------------------------
// Params / helpers
// -----------------------------------------------------------------------------

// codec2Params управляет минимальным размером блока и максимально допустимой "шероховатостью" (энергией градиента).
type codec2Params struct {
	minBlock           int   // минимальный размер стороны блока
	maxGrad            int32 // максимально допустимая энергия градиента для "листа"
	colorTol           int   // допуск по цвету для слияния палитры (в значениях канала)
	patternGainPercent int   // во сколько раз ошибка паттерна должна быть меньше solid (например: 2 = лучше на 50%)
}

type leafColor struct {
	c color.RGBA
}

type leafStats struct {
	total         int // всего листовых блоков
	patternBetter int // для скольких двухцветный паттерн даёт меньшую ошибку
}

// patternLeaf описывает лист, который будет кодироваться как двухцветный паттерн
// (foreground/background), без привязки к конкретной палитре. Палитра по-прежнему
// общая — оба цвета потом так же попадают в общий buildPalettes().
type patternLeaf struct {
	fg  color.RGBA
	bg  color.RGBA
	thr int32
}

type leafJob struct {
	x, y, w, h int
	pal        int // индекс палитры
	idx        int // локальный индекс цвета внутри палитры
}

// leafType описывает тип листа: однородный цвет или паттерн
type leafType uint8

const (
	leafTypeSolid leafType = iota
	leafTypePattern
)

// leafRef описывает ссылку листа на палитру и индексы цветов.
// typ определяет, является ли лист однородным (solid) или паттерн-листом.
// Для leafTypeSolid используется поле idx (один локальный индекс цвета).
// Для leafTypePattern будут задействованы fg/bg (два локальных индекса),
// pal остаётся общей привязкой к палитре.
type leafRef struct {
	typ leafType

	pal uint16
	idx uint8 // для solid-листа

	fg uint8 // локальный индекс fg для pattern-листа
	bg uint8 // локальный индекс bg для pattern-листа

	thr int32 // порог яркости для паттерн-листа (используется только энкодером)
}

// palette описывает одну локальную палитру (до 256 цветов).
type palette struct {
	colors  []color.RGBA
	buckets map[int][]int // bucketKey -> indices in colors, used only during buildPalettes
}

const (
	bucketStepY     = 16 // ~256/16 = 16 buckets by luma
	bucketStepCo    = 32 // coarser for chroma
	bucketStepCg    = 32
	bucketOffsetYCo = 512 // shift Co/Cg to positive range with margin
)

func paletteBucketCoords(c color.RGBA) (by, bco, bcg int) {
	y, co, cg := RgbToYCoCg(c)

	if y < 0 {
		y = 0
	} else if y > 255 {
		y = 255
	}

	by = int(y) / bucketStepY
	bco = int((co + bucketOffsetYCo) / bucketStepCo)
	bcg = int((cg + bucketOffsetYCo) / bucketStepCg)
	return
}

func paletteBucketKey(by, bco, bcg int) int {
	return (by << 16) | (bco << 8) | bcg
}

func (p *palette) addColor(c color.RGBA) int {
	idx := len(p.colors)
	p.colors = append(p.colors, c)
	if p.buckets != nil {
		by, bco, bcg := paletteBucketCoords(c)
		k := paletteBucketKey(by, bco, bcg)
		p.buckets[k] = append(p.buckets[k], idx)
	}
	return idx
}

func (p *palette) findCloseIndex(c color.RGBA, tol int) int {
	if len(p.colors) == 0 {
		return -1
	}
	if p.buckets == nil {
		// Fallback to linear search (should not happen during buildPalettes).
		for i, pc := range p.colors {
			if CloseColor(c, pc, tol) {
				return i
			}
		}
		return -1
	}

	by, bco, bcg := paletteBucketCoords(c)

	for dy := -1; dy <= 1; dy++ {
		for dc := -1; dc <= 1; dc++ {
			for dg := -1; dg <= 1; dg++ {
				// Check this bucket and its 26 neighbors (3x3x3 cube).
				ny := by + dy
				nco := bco + dc
				ncg := bcg + dg
				k := paletteBucketKey(ny, nco, ncg)
				indices := p.buckets[k]
				for _, idx := range indices {
					if CloseColor(c, p.colors[idx], tol) {
						return idx
					}
				}
			}
		}
	}
	return -1
}

type stripeInfo struct {
	x, y int
	w, h int
}

// nodeSplit описывает разбиение блока на две части.
type nodeSplit struct {
	x1, y1, w1, h1 int
	x2, y2, w2, h2 int
}

// splitNode делит блок (x,y,w,h) на две части.
// preferHorizontal определяет, какую ось пробовать первой:
//   - true: сначала делим по ширине, потом по высоте (если нельзя по ширине);
//   - false: сначала по высоте, потом по ширине.
func splitNode(x, y, w, h int, preferHorizontal bool) (nodeSplit, bool) {
	if w <= 1 && h <= 1 {
		return nodeSplit{}, false
	}

	if preferHorizontal {
		if w > 1 {
			w1 := w / 2
			w2 := w - w1
			return nodeSplit{
				x1: x, y1: y, w1: w1, h1: h,
				x2: x + w1, y2: y, w2: w2, h2: h,
			}, true
		}
		if h > 1 {
			h1 := h / 2
			h2 := h - h1
			return nodeSplit{
				x1: x, y1: y, w1: w, h1: h1,
				x2: x, y2: y + h1, w2: w, h2: h2,
			}, true
		}
	} else {
		if h > 1 {
			h1 := h / 2
			h2 := h - h1
			return nodeSplit{
				x1: x, y1: y, w1: w, h1: h1,
				x2: x, y2: y + h1, w2: w, h2: h2,
			}, true
		}
		if w > 1 {
			w1 := w / 2
			w2 := w - w1
			return nodeSplit{
				x1: x, y1: y, w1: w1, h1: h,
				x2: x + w1, y2: y, w2: w2, h2: h,
			}, true
		}
	}

	return nodeSplit{}, false
}

// splitStripes разбивает изображение на не более чем maxStripes полос
// вдоль более длинной стороны: для "высокого" кадра полосы идут по высоте,
// для "широкого" — по ширине. Каждая полоса описывается прямоугольником.
func splitStripes(totalW, totalH, minBlock int, maxStripes int) []stripeInfo {
	if totalW <= 0 || totalH <= 0 {
		return []stripeInfo{{x: 0, y: 0, w: totalW, h: totalH}}
	}

	// Выбираем направление разбиения: по более длинной стороне.
	splitHorizontally := totalH >= totalW // true => режем по высоте, false => по ширине
	var axisLen int
	if splitHorizontally {
		axisLen = totalH
	} else {
		axisLen = totalW
	}

	// Максимально допустимое количество полос.
	n := maxStripes
	if n < 1 {
		n = 1
	}

	// Не имеет смысла делать полос больше, чем количество минимальных блоков по выбранной оси.
	maxByAxis := axisLen / minBlock
	if maxByAxis < 1 {
		maxByAxis = 1
	}
	if n > maxByAxis {
		n = maxByAxis
	}

	// Одна полоса — весь кадр.
	if n == 1 {
		return []stripeInfo{{x: 0, y: 0, w: totalW, h: totalH}}
	}

	// дальше код такой же, только без encoderStripes
	base := axisLen / n
	rem := axisLen % n
	stripes := make([]stripeInfo, 0, n)

	cur := 0
	for i := 0; i < n; i++ {
		seg := base
		if i < rem {
			seg++
		}
		if splitHorizontally {
			stripes = append(stripes, stripeInfo{
				x: 0,
				y: cur,
				w: totalW,
				h: seg,
			})
		} else {
			stripes = append(stripes, stripeInfo{
				x: cur,
				y: 0,
				w: seg,
				h: totalH,
			})
		}
		cur += seg
	}
	return stripes
}

// ExpMap преобразует вход x из диапазона [inMin..inMax]
// в выход [outMin..outMax] по экспоненциальной кривой.
// k — крутизна (чем больше, тем резче рост в конце).
func ExpMap(x, inMin, inMax, outMin, outMax int, k float64) int {
	if inMax == inMin {
		return outMin
	}

	t := float64(x-inMin) / float64(inMax-inMin) // корректная нормализация 0..1
	curved := (math.Exp(k*t) - 1) / (math.Exp(k) - 1)
	return outMin + int(curved*float64(outMax-outMin))
}

// analyzeBlock computes the total, border, and inner energy (luma spread) and average color in a single scan.
func analyzeBlock(img *image.YCbCr, luma []int16, x, y, w, h int) (totalEnergy int32, borderEnergy int32, innerEnergy int32, avg color.RGBA) {
	b := img.Bounds()
	wImg := b.Dx()

	var minL int32 = 255
	var maxL int32 = 0

	var minLBorder int32 = 255
	var maxLBorder int32 = 0
	var minLInner int32 = 255
	var maxLInner int32 = 0

	var haveBorder, haveInner bool

	var sumY, sumCb, sumCr int64
	var count int64

	minX := b.Min.X
	minY := b.Min.Y

	for yy := 0; yy < h; yy++ {
		py := y + yy
		if py < b.Min.Y || py >= b.Max.Y {
			continue
		}
		for xx := 0; xx < w; xx++ {
			px := x + xx
			if px < b.Min.X || px >= b.Max.X {
				continue
			}
			idx := (py-minY)*wImg + (px - minX)
			l := int32(luma[idx])

			if l < minL {
				minL = l
			}
			if l > maxL {
				maxL = l
			}

			isBorder := xx == 0 || yy == 0 || xx == w-1 || yy == h-1
			if isBorder {
				if l < minLBorder {
					minLBorder = l
				}
				if l > maxLBorder {
					maxLBorder = l
				}
				haveBorder = true
			} else {
				if l < minLInner {
					minLInner = l
				}
				if l > maxLInner {
					maxLInner = l
				}
				haveInner = true
			}

			yIdx := (py-minY)*img.YStride + (px - minX)
			sumY += int64(img.Y[yIdx])
			sumCb += int64(img.Cb[yIdx])
			sumCr += int64(img.Cr[yIdx])
			count++
		}
	}

	if count == 0 {
		return 0, 0, 0, color.RGBA{0, 0, 0, 255}
	}

	totalEnergy = maxL - minL
	if haveBorder {
		borderEnergy = maxLBorder - minLBorder
	}
	if haveInner {
		innerEnergy = maxLInner - minLInner
	}

	avgY := uint8(sumY / count)
	avgCb := uint8(sumCb / count)
	avgCr := uint8(sumCr / count)
	avgYC := color.YCbCr{Y: avgY, Cb: avgCb, Cr: avgCr}
	avg = color.RGBAModel.Convert(avgYC).(color.RGBA)
	avg.A = 255
	return totalEnergy, borderEnergy, innerEnergy, avg
}

// collectRegionColorsStriped запускает collectRegionColors независимо для нескольких вертикальных полос
// и склеивает результат. Каждая полоса использует свою локальную форму дерева (pattern) и цвета листьев.
func collectRegionColorsStriped(
	img *image.YCbCr,
	luma []int16,
	params codec2Params,
	maxStripes int,
) ([]leafColor, []patternLeaf, []leafType, []bool, leafStats, error) {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()

	stripes := splitStripes(w, h, params.minBlock, maxStripes)
	if len(stripes) == 1 {
		var leaves []leafColor
		var pattLeaves []patternLeaf
		var modes []leafType
		var pattern []bool
		var stats leafStats
		if err := collectRegionColors(img, luma, 0, 0, w, h, params, &leaves, &pattLeaves, &modes, &pattern, &stats); err != nil {
			return nil, nil, nil, nil, leafStats{}, err
		}
		return leaves, pattLeaves, modes, pattern, stats, nil
	}

	type stripeResult struct {
		leaves     []leafColor
		pattLeaves []patternLeaf
		modes      []leafType
		pattern    []bool
		stats      leafStats
		err        error
	}

	results := make([]stripeResult, len(stripes))

	// Решаем, есть ли смысл плодить горутины.
	// Для небольших изображений и малых полос всё делаем последовательно.
	totalPixels := w * h
	maxWorkers := runtime.NumCPU()
	const minPixelsForParallel = 512 * 512

	useGoroutines := totalPixels >= minPixelsForParallel || len(stripes) > maxWorkers

	if useGoroutines {
		var wg sync.WaitGroup
		for i, s := range stripes {
			wg.Add(1)
			go func(i int, s stripeInfo) {
				defer wg.Done()
				var leaves []leafColor
				var pattLeaves []patternLeaf
				var modes []leafType
				var pattern []bool
				var stats leafStats
				err := collectRegionColors(img, luma, s.x, s.y, s.w, s.h, params, &leaves, &pattLeaves, &modes, &pattern, &stats)
				results[i] = stripeResult{
					leaves:     leaves,
					pattLeaves: pattLeaves,
					modes:      modes,
					pattern:    pattern,
					stats:      stats,
					err:        err,
				}
			}(i, s)
		}
		wg.Wait()
	} else {
		// Последовательная обработка всех полос без горутин.
		for i, s := range stripes {
			var leaves []leafColor
			var pattLeaves []patternLeaf
			var modes []leafType
			var pattern []bool
			var stats leafStats
			err := collectRegionColors(img, luma, s.x, s.y, s.w, s.h, params, &leaves, &pattLeaves, &modes, &pattern, &stats)
			results[i] = stripeResult{
				leaves:     leaves,
				pattLeaves: pattLeaves,
				modes:      modes,
				pattern:    pattern,
				stats:      stats,
				err:        err,
			}
		}
	}

	// Проверяем ошибки и склеиваем результаты в порядке полос сверху вниз.
	var totalLeaves int
	var totalPattLeaves int
	var totalModes int
	var totalPattern int
	var totalStats leafStats
	for _, r := range results {
		if r.err != nil {
			return nil, nil, nil, nil, leafStats{}, r.err
		}
		totalLeaves += len(r.leaves)
		totalPattLeaves += len(r.pattLeaves)
		totalModes += len(r.modes)
		totalPattern += len(r.pattern)
		totalStats.total += r.stats.total
		totalStats.patternBetter += r.stats.patternBetter
	}

	leaves := make([]leafColor, 0, totalLeaves)
	pattLeaves := make([]patternLeaf, 0, totalPattLeaves)
	modes := make([]leafType, 0, totalModes)
	pattern := make([]bool, 0, totalPattern)
	for _, r := range results {
		leaves = append(leaves, r.leaves...)
		pattLeaves = append(pattLeaves, r.pattLeaves...)
		modes = append(modes, r.modes...)
		pattern = append(pattern, r.pattern...)
	}

	return leaves, pattLeaves, modes, pattern, totalStats, nil
}

// computeFG_BG вычисляет два усреднённых цвета (fg/bg) для блока,
// разделяя пиксели по порогу яркости thr.
func computeFG_BG(img *image.YCbCr, luma []int16, x, y, w, h int) (fg, bg color.RGBA, thr int32, ok bool) {
	b := img.Bounds()
	wImg := b.Dx()

	minX := b.Min.X
	minY := b.Min.Y

	var sumL int64
	var count int64
	for yy := 0; yy < h; yy++ {
		py := y + yy
		if py < b.Min.Y || py >= b.Max.Y {
			continue
		}
		for xx := 0; xx < w; xx++ {
			px := x + xx
			if px < b.Min.X || px >= b.Max.X {
				continue
			}
			idx := (py-minY)*wImg + (px - minX)
			sumL += int64(luma[idx])
			count++
		}
	}
	if count == 0 {
		return color.RGBA{0, 0, 0, 255}, color.RGBA{0, 0, 0, 255}, 0, false
	}
	thr = int32(sumL / count)

	var fgY, fgCb, fgCr, bgY, bgCb, bgCr int64
	var fgCount, bgCount int64

	for yy := 0; yy < h; yy++ {
		py := y + yy
		if py < b.Min.Y || py >= b.Max.Y {
			continue
		}
		for xx := 0; xx < w; xx++ {
			px := x + xx
			if px < b.Min.X || px >= b.Max.X {
				continue
			}
			idx := (py-minY)*wImg + (px - minX)
			l := int32(luma[idx])

			yIdx := (py-minY)*img.YStride + (px - minX)
			Y := img.Y[yIdx]
			Cb := img.Cb[yIdx]
			Cr := img.Cr[yIdx]

			if l >= thr {
				fgY += int64(Y)
				fgCb += int64(Cb)
				fgCr += int64(Cr)
				fgCount++
			} else {
				bgY += int64(Y)
				bgCb += int64(Cb)
				bgCr += int64(Cr)
				bgCount++
			}
		}
	}

	if fgCount == 0 {
		fgCount = 1
	}
	if bgCount == 0 {
		bgCount = 1
	}

	fgYC := color.YCbCr{
		Y:  uint8(fgY / fgCount),
		Cb: uint8(fgCb / fgCount),
		Cr: uint8(fgCr / fgCount),
	}
	bgYC := color.YCbCr{
		Y:  uint8(bgY / bgCount),
		Cb: uint8(bgCb / bgCount),
		Cr: uint8(bgCr / bgCount),
	}
	fg = color.RGBAModel.Convert(fgYC).(color.RGBA)
	bg = color.RGBAModel.Convert(bgYC).(color.RGBA)
	fg.A = 255
	bg.A = 255
	return fg, bg, thr, true
}

func pickLeafModel(
	img *image.YCbCr,
	luma []int16,
	x, y, w, h int,
	avg color.RGBA,
	params codec2Params,
	stats *leafStats,
) (leafType, color.RGBA, patternLeaf) {
	b := img.Bounds()
	wImg := b.Dx()

	minX := b.Min.X
	minY := b.Min.Y

	avgL := lumaFromRGB(avg.R, avg.G, avg.B)

	// Ошибка одноцветной модели по яркости.
	var errSolid int64
	for yy := 0; yy < h; yy++ {
		py := y + yy
		if py < b.Min.Y || py >= b.Max.Y {
			continue
		}
		for xx := 0; xx < w; xx++ {
			px := x + xx
			if px < b.Min.X || px >= b.Max.X {
				continue
			}
			idx := (py-minY)*wImg + (px - minX)
			l := int32(luma[idx])
			dl := l - avgL
			if dl < 0 {
				dl = -dl
			}
			errSolid += int64(dl)
		}
	}

	fg, bg, thr, ok := computeFG_BG(img, luma, x, y, w, h)
	if !ok {
		if stats != nil {
			stats.total++
		}
		return leafTypeSolid, avg, patternLeaf{}
	}

	// Precompute luma for fg and bg.
	fgL := lumaFromRGB(fg.R, fg.G, fg.B)
	bgL := lumaFromRGB(bg.R, bg.G, bg.B)

	// Ошибка двухцветной модели по яркости.
	var errPattern int64
	for yy := 0; yy < h; yy++ {
		py := y + yy
		if py < b.Min.Y || py >= b.Max.Y {
			continue
		}
		for xx := 0; xx < w; xx++ {
			px := x + xx
			if px < b.Min.X || px >= b.Max.X {
				continue
			}
			idx := (py-minY)*wImg + (px - minX)
			l := int32(luma[idx])
			var aproxL int32
			if l >= thr {
				aproxL = fgL
			} else {
				aproxL = bgL
			}
			dl := l - aproxL
			if dl < 0 {
				dl = -dl
			}
			errPattern += int64(dl)
		}
	}

	// Требуемое относительное улучшение ошибки в процентах.
	// patternGainPercent = 30 => errPattern < 0.7 * errSolid.
	gain := params.patternGainPercent
	if gain < 0 {
		gain = 0
	}
	if gain > 99 {
		gain = 99
	}

	// errPattern < (1 - gain/100) * errSolid
	if stats != nil {
		stats.total++
		if errPattern*100 < errSolid*int64(100-gain) {
			stats.patternBetter++
		}
	}

	if errPattern*100 < errSolid*int64(100-gain) {
		return leafTypePattern, avg, patternLeaf{fg: fg, bg: bg, thr: thr}
	}
	return leafTypeSolid, avg, patternLeaf{}
}

func collectRegionColors(
	img *image.YCbCr,
	luma []int16,
	x, y, w, h int,
	params codec2Params,
	leaves *[]leafColor,
	pattLeaves *[]patternLeaf,
	modes *[]leafType,
	pattern *[]bool,
	stats *leafStats,
) error {
	// Логика условий листа должна совпадать с encodeRegion / decodeRegionJobs.
	// Для каждого узла мы должны записать РОВНО ОДИН бит в pattern:
	// true  = лист
	// false = внутренний узел

	// 1) Базовое условие по размеру — сразу лист.
	if w <= params.minBlock && h <= params.minBlock {
		energy, _, _, avg := analyzeBlock(img, luma, x, y, w, h)

		// Если блок маленький, но всё ещё "жесткий" — насильно делим ещё, пока можем.
		if energy > params.maxGrad && (w > 1 || h > 1) {
			*pattern = append(*pattern, false) // внутренний узел

			sp, ok := splitNode(x, y, w, h, w >= h)
			if !ok {
				return fmt.Errorf("collectRegionColors: internal node with non-divisible geometry (%d×%d)", w, h)
			}

			if err := collectRegionColors(img, luma, sp.x1, sp.y1, sp.w1, sp.h1, params, leaves, pattLeaves, modes, pattern, stats); err != nil {
				return err
			}
			if err := collectRegionColors(img, luma, sp.x2, sp.y2, sp.w2, sp.h2, params, leaves, pattLeaves, modes, pattern, stats); err != nil {
				return err
			}
			return nil
		}

		// иначе — обычный лист
		*pattern = append(*pattern, true)
		mode, solid, patt := pickLeafModel(img, luma, x, y, w, h, avg, params, stats)
		if mode == leafTypePattern {
			*pattLeaves = append(*pattLeaves, patt)
		}
		*modes = append(*modes, mode)
		*leaves = append(*leaves, leafColor{c: solid})
		return nil
	}

	// 2) Оцениваем "шероховатость" блока и одновременно считаем средний цвет.
	energy, borderEnergy, innerEnergy, avg := analyzeBlock(img, luma, x, y, w, h)
	if energy == 0 {
		// Идеально ровный блок - лист.
		*pattern = append(*pattern, true)
		mode, solid, patt := pickLeafModel(img, luma, x, y, w, h, avg, params, stats)
		if mode == leafTypePattern {
			*pattLeaves = append(*pattLeaves, patt)
		}
		*modes = append(*modes, mode)
		*leaves = append(*leaves, leafColor{c: solid})
		return nil
	}

	// Определяем, не является ли блок "краевым": когда большая часть контраста
	// сосредоточена именно на границе блока, а внутри относительно ровно.
	edgeDominated := false
	if borderEnergy > 0 && innerEnergy > 0 {
		// Если энергия на границе заметно превышает внутреннюю, и блок ещё достаточно крупный,
		// имеет смысл делить его дальше, чтобы граница объекта проходила по более мелким блокам.
		if borderEnergy >= innerEnergy*2 && (w > params.minBlock*2 || h > params.minBlock*2) {
			edgeDominated = true
		}
	}

	if !edgeDominated && energy <= params.maxGrad && w <= 4*params.minBlock && h <= 4*params.minBlock {
		// Блок достаточно гладкий И уже не гигантский — лист.
		*pattern = append(*pattern, true)
		mode, solid, patt := pickLeafModel(img, luma, x, y, w, h, avg, params, stats)
		if mode == leafTypePattern {
			*pattLeaves = append(*pattLeaves, patt)
		}
		*modes = append(*modes, mode)
		*leaves = append(*leaves, leafColor{c: solid})
		return nil
	}

	// 3) Здесь блок считается "пёстрым" и мы хотим его делить.
	// Но сначала убеждаемся, что реально можем разделить по какой-то стороне.
	if w >= h && w/2 >= params.minBlock {
		// Внутренний узел: делим по ширине.
		*pattern = append(*pattern, false)

		sp, ok := splitNode(x, y, w, h, true)
		if !ok {
			return fmt.Errorf("collectRegionColors: internal node with non-divisible geometry (%d×%d)", w, h)
		}

		if err := collectRegionColors(img, luma, sp.x1, sp.y1, sp.w1, sp.h1, params, leaves, pattLeaves, modes, pattern, stats); err != nil {
			return err
		}
		if err := collectRegionColors(img, luma, sp.x2, sp.y2, sp.w2, sp.h2, params, leaves, pattLeaves, modes, pattern, stats); err != nil {
			return err
		}
		return nil
	}

	// Пробуем делить по высоте.
	sp, ok := splitNode(x, y, w, h, false)
	if !ok || sp.h1 < params.minBlock {
		// Слишком маленький для деления блок — принудительно лист.
		*pattern = append(*pattern, true)
		_, _, _, avg := analyzeBlock(img, luma, x, y, w, h)
		mode, solid, patt := pickLeafModel(img, luma, x, y, w, h, avg, params, stats)
		if mode == leafTypePattern {
			*pattLeaves = append(*pattLeaves, patt)
		}
		*modes = append(*modes, mode)
		*leaves = append(*leaves, leafColor{c: solid})
		return nil
	}

	// Внутренний узел: делим по высоте.
	*pattern = append(*pattern, false)
	if err := collectRegionColors(img, luma, sp.x1, sp.y1, sp.w1, sp.h1, params, leaves, pattLeaves, modes, pattern, stats); err != nil {
		return err
	}
	if err := collectRegionColors(img, luma, sp.x2, sp.y2, sp.w2, sp.h2, params, leaves, pattLeaves, modes, pattern, stats); err != nil {
		return err
	}

	return nil
}

// buildPalettes строит набор локальных палитр и ссылок на цвета для каждого листа.
// Мы накапливаем цвета в текущей палитре до тех пор, пока не появится новый
// "достаточно отличный" цвет и в палитре уже 256 уникальных значений — тогда
// заводим новую палитру. Таким образом, одна палитра хранит не более 256 цветов,
// а цвета внутри неё сгруппированы по близости (в пространстве YCoCg).
//
// На этом шаге:
//   - для leafTypeSolid в палитру попадает один цвет (leaves[i].c),
//   - для leafTypePattern в палитру попадают оба цвета (fg/bg из pattLeaves),
//   - leafRef.typ заполняется корректно,
//   - leafRef.idx всегда указывает на "основной" цвет листа:
//   - для solid — единственный цвет,
//   - для pattern — сейчас берём fg-индекс (для совместимости с текущим кодом).
func buildPalettes(leaves []leafColor, pattLeaves []patternLeaf, modes []leafType, params codec2Params) ([]palette, []leafRef, error) {
	if len(leaves) == 0 {
		return nil, nil, nil
	}
	if len(modes) != len(leaves) {
		return nil, nil, fmt.Errorf("buildPalettes: len(modes)=%d != len(leaves)=%d", len(modes), len(leaves))
	}

	var pals []palette
	pals = append(pals, palette{
		colors:  make([]color.RGBA, 0, 256),
		buckets: make(map[int][]int),
	})

	refs := make([]leafRef, len(leaves))
	curPalID := 0
	pattIdx := 0

	for i, mode := range modes {
		p := &pals[curPalID]

		switch mode {
		case leafTypeSolid:
			// Обычный одноцветный лист: используем усреднённый цвет leaves[i].c.
			c := leaves[i].c

			// Пытаемся найти "близкий" цвет в текущей палитре через бакеты.
			found := p.findCloseIndex(c, params.colorTol)
			if found < 0 {
				// Новый цвет. Если текущая палитра уже содержит 256 уникальных цветов,
				// заводим новую палитру и переключаемся на неё.
				if len(p.colors) >= 256 {
					pals = append(pals, palette{
						colors:  make([]color.RGBA, 0, 256),
						buckets: make(map[int][]int),
					})
					curPalID++
					p = &pals[curPalID]
				}
				found = p.addColor(c)
			}

			refs[i] = leafRef{
				typ: leafTypeSolid,
				pal: uint16(curPalID),
				idx: uint8(found),
			}

		case leafTypePattern:
			// Паттерн-лист: два цвета (fg/bg). Для будущего формата оба цвета будут
			// ссылаться на одну и ту же палитру, поэтому следим, чтобы внутри
			// текущей палитры было не менее двух свободных слотов.
			if pattIdx >= len(pattLeaves) {
				return nil, nil, fmt.Errorf("buildPalettes: pattIdx %d out of range (len=%d)", pattIdx, len(pattLeaves))
			}
			pl := pattLeaves[pattIdx]
			pattIdx++

			// Если в палитре осталось меньше двух мест, создаём новую, чтобы fg/bg
			// гарантированно поместились в одну палитру.
			if len(p.colors) > 254 {
				pals = append(pals, palette{
					colors:  make([]color.RGBA, 0, 256),
					buckets: make(map[int][]int),
				})
				curPalID++
				p = &pals[curPalID]
			}

			// Функция поиска/добавления цвета в текущую палитру.
			findOrAdd := func(c color.RGBA) int {
				if idx := p.findCloseIndex(c, params.colorTol); idx >= 0 {
					return idx
				}
				// Новый цвет.
				if len(p.colors) >= 256 {
					// Защита от логических ошибок выше: сюда попадать не должны.
					panic("buildPalettes: palette overflow for pattern leaf")
				}
				return p.addColor(c)
			}

			fgIdx := findOrAdd(pl.fg)
			bgIdx := findOrAdd(pl.bg)

			refs[i] = leafRef{
				typ: leafTypePattern,
				pal: uint16(curPalID),
				// idx используется текущим кодом encodeRegion/leafLocal как "основной" цвет листа,
				// чтобы не ломать формат, кладём туда fg-индекс.
				idx: uint8(fgIdx),
				fg:  uint8(fgIdx),
				bg:  uint8(bgIdx),
				thr: pl.thr,
			}

		default:
			return nil, nil, fmt.Errorf("buildPalettes: unknown leafType %d at index %d", mode, i)
		}
	}

	// На всякий случай проверим, что мы прошли все pattern-листы.
	if pattIdx != len(pattLeaves) {
		return nil, nil, fmt.Errorf("buildPalettes: used %d pattern leaves, but pattLeaves has %d", pattIdx, len(pattLeaves))
	}

	// Дополнительный шаг: внутри каждой палитры сортируем цвета по YCoCg,
	// а индексы во всех ссылках leafRef перенумеровываем согласно новой перестановке.
	for pid := range pals {
		colors := pals[pid].colors
		if len(colors) <= 1 {
			continue
		}

		// Подготовим структуру с исходным индексом и координатами YCoCg.
		type ycColor struct {
			oldIdx    int
			c         color.RGBA
			y, co, cg int32
		}
		ycs := make([]ycColor, len(colors))
		for i, c := range colors {
			y, co, cg := RgbToYCoCg(c)
			ycs[i] = ycColor{oldIdx: i, c: c, y: y, co: co, cg: cg}
		}

		// Сортируем по яркости, затем по цветовым компонентам в YCoCg.
		sort.Slice(ycs, func(i, j int) bool {
			if ycs[i].y != ycs[j].y {
				return ycs[i].y < ycs[j].y
			}
			if ycs[i].co != ycs[j].co {
				return ycs[i].co < ycs[j].co
			}
			return ycs[i].cg < ycs[j].cg
		})

		// Строим отображение oldIdx -> newIdx и новую палитру цветов.
		inv := make([]int, len(colors))
		newColors := make([]color.RGBA, len(colors))
		for newIdx, yc := range ycs {
			newColors[newIdx] = yc.c
			inv[yc.oldIdx] = newIdx
		}
		pals[pid].colors = newColors

		// Обновляем индексы во всех ссылках leafRef, которые ссылаются на эту палитру.
		for i := range refs {
			if int(refs[i].pal) != pid {
				continue
			}
			switch refs[i].typ {
			case leafTypeSolid:
				old := int(refs[i].idx)
				if old < 0 || old >= len(inv) {
					return nil, nil, fmt.Errorf("buildPalettes: solid index %d out of range for palette %d", old, pid)
				}
				refs[i].idx = uint8(inv[old])
			case leafTypePattern:
				oldFg := int(refs[i].fg)
				oldBg := int(refs[i].bg)
				if oldFg < 0 || oldFg >= len(inv) || oldBg < 0 || oldBg >= len(inv) {
					return nil, nil, fmt.Errorf("buildPalettes: pattern indices (%d,%d) out of range for palette %d", oldFg, oldBg, pid)
				}
				newFg := inv[oldFg]
				newBg := inv[oldBg]
				refs[i].fg = uint8(newFg)
				refs[i].bg = uint8(newBg)
				// idx хранит "основной" цвет листа — обновляем его так же, как fg.
				refs[i].idx = uint8(newFg)
			}
		}
		// После сортировки бакеты становятся невалидны и больше не нужны.
		pals[pid].buckets = nil
	}

	return pals, refs, nil
}

// -----------------------------------------------------------------------------
// Recursive region encode/decode
// -----------------------------------------------------------------------------

// encodeRegion рекурсивно кодирует прямоугольник (x,y,w,h) по заранее вычисленному дереву (pattern).
// pattern: []bool, где каждый бит указывает, является ли данный узел листом (true) или внутренним (false).
func encodeRegion(
	img *image.YCbCr,
	luma []int16,
	x, y, w, h int,
	params codec2Params,
	bw *BitWriter,
	refs []leafRef,
	leafPos *int,
	pattern []bool,
	patPos *int,
	leafBuf *bytes.Buffer,
	patBW *BitWriter,
) error {
	// На каждом узле читаем заранее принятие решение: лист или внутренний узел.
	if *patPos >= len(pattern) {
		return fmt.Errorf("encodeRegion: pattern overflow")
	}
	isLeaf := pattern[*patPos]
	*patPos++

	if isLeaf {
		// Лист: берём следующую ссылку leafRef и кодируем её в соответствии с типом.
		if *leafPos >= len(refs) {
			return fmt.Errorf("encodeRegion: leaf index overflow")
		}
		r := refs[*leafPos]
		*leafPos++

		// помечаем лист в дереве
		if err := bw.WriteBit(true); err != nil {
			return err
		}

		switch r.typ {
		case leafTypeSolid:
			// Однородный лист: тип 0, один локальный индекс цвета.
			if err := bw.WriteBit(false); err != nil { // 0 = leafTypeSolid
				return err
			}
			// В поток индексов пишем только локальный индекс цвета.
			if err := leafBuf.WriteByte(r.idx); err != nil {
				return err
			}
			return nil

		case leafTypePattern:
			// Паттерн-лист: тип 1, два локальных индекса (fg/bg) и w*h бит паттерна.
			if err := bw.WriteBit(true); err != nil { // 1 = leafTypePattern
				return err
			}
			// В поток индексов пишем только локальные индексы fg/bg.
			if err := leafBuf.WriteByte(r.fg); err != nil {
				return err
			}
			if err := leafBuf.WriteByte(r.bg); err != nil {
				return err
			}

			// Используем заранее сохранённый порог яркости для этого листа.
			thr := r.thr

			// Cached geometry and pixel pointers for the pattern generation loop.
			b := img.Bounds()
			wImg := b.Dx()
			minX := b.Min.X
			minY := b.Min.Y

			for yy := 0; yy < h; yy++ {
				for xx := 0; xx < w; xx++ {
					px := x + xx
					py := y + yy
					if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
						// Вне исходного изображения считаем фоном.
						if err := patBW.WriteBit(false); err != nil {
							return err
						}
						continue
					}
					idx := (py-minY)*wImg + (px - minX)
					l := int32(luma[idx])
					if l >= thr {
						if err := patBW.WriteBit(true); err != nil {
							return err
						}
					} else {
						if err := patBW.WriteBit(false); err != nil {
							return err
						}
					}
				}
			}
			return nil

		default:
			return fmt.Errorf("encodeRegion: unknown leaf type %d", r.typ)
		}
	}

	// Внутренний узел: пишем бит 0 и делим блок так же, как в collectRegionColors/decodeRegionJobs.
	if err := bw.WriteBit(false); err != nil { // 0 = внутренний узел
		return err
	}

	sp, ok := splitNode(x, y, w, h, w >= h)
	if !ok {
		// Если сюда попали, значит pattern пометил внутренний узел для блока,
		// который уже нельзя делить геометрически — это логическая ошибка.
		return fmt.Errorf("encodeRegion: internal node with non-divisible geometry (%d×%d)", w, h)
	}

	if err := encodeRegion(img, luma, sp.x1, sp.y1, sp.w1, sp.h1, params, bw, refs, leafPos, pattern, patPos, leafBuf, patBW); err != nil {
		return err
	}
	if err := encodeRegion(img, luma, sp.x2, sp.y2, sp.w2, sp.h2, params, bw, refs, leafPos, pattern, patPos, leafBuf, patBW); err != nil {
		return err
	}
	return nil
}

// decodeRegionJobs зеркален decodeRegion, но вместо отрисовки листьев
// собирает список "работ" по заливке блоков цветом. Это позволяет
// считать битстрим последовательно, а рисовать блоки — параллельно.
func decodeRegionJobs(
	x, y, w, h int,
	params codec2Params,
	br *BitReader,
	patBr *BitReader,
	dst *image.RGBA,
	palettes [][]color.RGBA,
	leafCounts []uint32,
	curPal *int,
	usedInPal *int,
	leafBytes []byte,
	leafPos *int,
	jobs *[]leafJob,
) error {
	isLeaf, err := br.ReadBit()
	if err != nil {
		return err
	}
	if isLeaf {
		// Читаем тип листа: 0 = однородный, 1 = паттерн.
		typeBit, err := br.ReadBit()
		if err != nil {
			return err
		}

		// Определяем, к какой палитре относится этот лист, по leafCounts.
		for *curPal < len(palettes) && *usedInPal >= int(leafCounts[*curPal]) {
			*curPal++
			*usedInPal = 0
		}
		if *curPal >= len(palettes) {
			*curPal = len(palettes) - 1
		}
		palID := *curPal

		if !typeBit {
			// Однородный лист: читаем один индекс и добавляем задачу заливки.
			if *leafPos >= len(leafBytes) {
				return fmt.Errorf("decodeRegionJobs: leaf index stream underflow")
			}
			b := leafBytes[*leafPos]
			*leafPos++

			local := int(b)
			if local < 0 || local >= len(palettes[palID]) {
				return fmt.Errorf("decodeRegionJobs: local color index %d out of range for palette %d", local, palID)
			}
			*jobs = append(*jobs, leafJob{
				x:   x,
				y:   y,
				w:   w,
				h:   h,
				pal: palID,
				idx: local,
			})
			*usedInPal++
			return nil
		}

		// Паттерн-лист: читаем два индекса fg/bg и w*h бит паттерна, сразу рисуем блок.
		if *leafPos+1 >= len(leafBytes) {
			return fmt.Errorf("decodeRegionJobs: pattern leaf index stream underflow")
		}
		fgIdx := int(leafBytes[*leafPos])
		bgIdx := int(leafBytes[*leafPos+1])
		*leafPos += 2

		if fgIdx < 0 || fgIdx >= len(palettes[palID]) || bgIdx < 0 || bgIdx >= len(palettes[palID]) {
			return fmt.Errorf("decodeRegionJobs: fg/bg index out of range for palette %d", palID)
		}
		fg := palettes[palID][fgIdx]
		bg := palettes[palID][bgIdx]

		bounds := dst.Bounds()
		for yy := 0; yy < h; yy++ {
			for xx := 0; xx < w; xx++ {
				bit, err := patBr.ReadBit()
				if err != nil {
					return fmt.Errorf("decodeRegionJobs: pattern bitstream underflow: %w", err)
				}
				px := x + xx
				py := y + yy
				if px < bounds.Min.X || py < bounds.Min.Y || px >= bounds.Max.X || py >= bounds.Max.Y {
					continue
				}
				if bit {
					dst.SetRGBA(px, py, fg)
				} else {
					dst.SetRGBA(px, py, bg)
				}
			}
		}
		*usedInPal++
		return nil
	}

	// Внутренний узел - делим так же, как в encodeRegion/collectRegionColors.
	sp, ok := splitNode(x, y, w, h, w >= h)
	if !ok {
		// Встретили внутренний узел, который уже нельзя делить по геометрии.
		return fmt.Errorf("decodeRegionJobs: internal node with non-divisible geometry (%d×%d)", w, h)
	}

	if err := decodeRegionJobs(sp.x1, sp.y1, sp.w1, sp.h1, params, br, patBr, dst, palettes, leafCounts, curPal, usedInPal, leafBytes, leafPos, jobs); err != nil {
		return err
	}
	if err := decodeRegionJobs(sp.x2, sp.y2, sp.w2, sp.h2, params, br, patBr, dst, palettes, leafCounts, curPal, usedInPal, leafBytes, leafPos, jobs); err != nil {
		return err
	}
	return nil
}

func paintLeafJobsParallel(dst *image.RGBA, palettes [][]color.RGBA, jobs []leafJob) {
	if len(jobs) == 0 {
		return
	}

	bounds := dst.Bounds()
	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}
	if len(jobs) < workers {
		workers = len(jobs)
	}

	var wg sync.WaitGroup
	chunk := (len(jobs) + workers - 1) / workers

	for wIdx := 0; wIdx < workers; wIdx++ {
		start := wIdx * chunk
		end := start + chunk
		if start >= len(jobs) {
			break
		}
		if end > len(jobs) {
			end = len(jobs)
		}

		wg.Add(1)
		go func(js []leafJob) {
			defer wg.Done()
			for _, job := range js {
				pal := palettes[job.pal]
				c := pal[job.idx]
				for yy := 0; yy < job.h; yy++ {
					for xx := 0; xx < job.w; xx++ {
						px := job.x + xx
						py := job.y + yy
						if px < bounds.Min.X || py < bounds.Min.Y || px >= bounds.Max.X || py >= bounds.Max.Y {
							continue
						}
						dst.SetRGBA(px, py, c)
					}
				}
			}
		}(jobs[start:end])
	}

	wg.Wait()
}

// -----------------------------------------------------------------------------
// BitWriter / BitReader
// -----------------------------------------------------------------------------

// BitWriter writes bits (MSB-first) into an underlying io.ByteWriter / bytes.Buffer.
type BitWriter struct {
	buf  *bytes.Buffer
	acc  byte
	nbit uint8 // сколько бит уже занято в acc (0..7)
}

// BitReader читает биты/байты из []byte.
type BitReader struct {
	data []byte
	pos  int   // индекс байта
	acc  byte  // текущий байт
	nbit uint8 // сколько бит уже прочитано из acc (0..8)
}

// -----------------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------------

var ErrInvalidMagic = errors.New("codec2: invalid magic")

func ValidateQuality(q int) int {
	if q < 1 {
		return 1
	}
	if q > 100 {
		return 100
	}
	return q
}

// ImageToYCbCr copies any image.Image into an *image.YCbCr (4:4:4) with bounds starting at (0,0).
func ImageToYCbCr(src image.Image) *image.YCbCr {
	b := src.Bounds()
	rect := image.Rect(0, 0, b.Dx(), b.Dy())
	dst := image.NewYCbCr(rect, image.YCbCrSubsampleRatio444)

	for y := 0; y < rect.Dy(); y++ {
		for x := 0; x < rect.Dx(); x++ {
			c := color.YCbCrModel.Convert(src.At(b.Min.X+x, b.Min.Y+y)).(color.YCbCr)
			i := y*dst.YStride + x
			dst.Y[i] = c.Y
			dst.Cb[i] = c.Cb
			dst.Cr[i] = c.Cr
		}
	}
	return dst
}

// buildLumaBuffer computes luma (0..255) for each pixel of an RGBA image
// and returns a flat slice indexed as (y - minY)*width + (x - minX).
func buildLumaBuffer(img *image.YCbCr) []int16 {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	if w <= 0 || h <= 0 {
		return nil
	}

	buf := make([]int16, w*h)

	minX := b.Min.X
	minY := b.Min.Y

	for yy := b.Min.Y; yy < b.Max.Y; yy++ {
		rowIdx := (yy - minY) * w
		for xx := b.Min.X; xx < b.Max.X; xx++ {
			yIdx := (yy-minY)*img.YStride + (xx - minX)
			Y := img.Y[yIdx]
			buf[rowIdx+(xx-minX)] = int16(Y)
		}
	}

	return buf
}

func WriteHeader(b *bytes.Buffer, quality int, w, h uint16, stripes uint8) error {
	// Header: magic(4) + width(uint16) + height(uint16) + quality(uint8) + stripes(uint8)
	if _, err := b.Write([]byte(magicCodec2)); err != nil {
		return err
	}

	if err := binary.Write(b, binary.BigEndian, w); err != nil {
		return err
	}
	if err := binary.Write(b, binary.BigEndian, h); err != nil {
		return err
	}
	if err := b.WriteByte(byte(quality)); err != nil {
		return err
	}
	if err := b.WriteByte(stripes); err != nil {
		return err
	}
	return nil
}

func ReadHeader(r *bytes.Reader) (quality, w, h, stripes int, err error) {
	// Read header.
	magic := make([]byte, len(magicCodec2))
	if _, err = r.Read(magic); err != nil {
		return
	}
	if string(magic) != magicCodec2 {
		err = ErrInvalidMagic
		return
	}

	var w16, h16 uint16
	if err = binary.Read(r, binary.BigEndian, &w16); err != nil {
		return
	}
	if err = binary.Read(r, binary.BigEndian, &h16); err != nil {
		return
	}
	qByte, e := r.ReadByte()
	if e != nil {
		err = e
		return
	}
	stripesByte, e := r.ReadByte()
	if e != nil {
		err = e
		return
	}

	quality = int(qByte)
	w, h = int(w16), int(h16)
	stripes = int(stripesByte)
	if stripes < 1 {
		stripes = 1
	}
	return
}

// CloseColor returns true if colors a and b are "close enough" (approximate match),
// но сравнение делается в пространстве YCoCg, а не в исходном RGB.
// tol интерпретируется как допуск по цветовым каналам (Co/Cg), а по яркости
// допускаем немного больший разброс.
func CloseColor(a, b color.RGBA, tol int) bool {
	ya, coa, cga := RgbToYCoCg(a)
	yb, cob, cgb := RgbToYCoCg(b)

	dY := Abs32(ya - yb)
	dCo := Abs32(coa - cob)
	dCg := Abs32(cga - cgb)

	// Яркость можно допускать чуть сильнее, чем цветовой сдвиг.
	yTol := int32(tol * 2)
	cTol := int32(tol)

	return dY <= yTol && dCo <= cTol && dCg <= cTol
}

// WriteBit writes a single bit.
func (bw *BitWriter) WriteBit(v bool) error {
	if v {
		bw.acc |= 1 << (7 - bw.nbit)
	}
	bw.nbit++
	if bw.nbit == 8 {
		if err := bw.buf.WriteByte(bw.acc); err != nil {
			return err
		}
		bw.acc = 0
		bw.nbit = 0
	}
	return nil
}

// WriteByte writes a full byte, respecting current bit alignment.
func (bw *BitWriter) WriteByte(b byte) error {
	// Если по байтовой границе, пишем напрямую.
	if bw.nbit == 0 {
		return bw.buf.WriteByte(b)
	}
	// Иначе — по битам.
	for i := 0; i < 8; i++ {
		bit := (b & (1 << (7 - i))) != 0
		if err := bw.WriteBit(bit); err != nil {
			return err
		}
	}
	return nil
}

// Flush дописывает хвостовой байт, если нужно.
func (bw *BitWriter) Flush() error {
	if bw.nbit == 0 {
		return nil
	}
	if err := bw.buf.WriteByte(bw.acc); err != nil {
		return err
	}
	bw.acc = 0
	bw.nbit = 0
	return nil
}

// ReadBit читает один бит.
func (br *BitReader) ReadBit() (bool, error) {
	if br.nbit == 0 {
		if br.pos >= len(br.data) {
			return false, io.EOF
		}
		br.acc = br.data[br.pos]
		br.pos++
	}
	bit := (br.acc & (1 << (7 - br.nbit))) != 0
	br.nbit++
	if br.nbit == 8 {
		br.nbit = 0
	}
	return bit, nil
}

// ReadByte читает байт, учитывая текущее битовое выравнивание.
func (br *BitReader) ReadByte() (byte, error) {
	if br.nbit == 0 {
		if br.pos >= len(br.data) {
			return 0, io.EOF
		}
		b := br.data[br.pos]
		br.pos++
		return b, nil
	}
	// Не по границе — собираем по битам.
	var b byte
	for i := 0; i < 8; i++ {
		bit, err := br.ReadBit()
		if err != nil {
			return 0, err
		}
		if bit {
			b |= 1 << (7 - i)
		}
	}
	return b, nil
}

func SmoothEdges(src *image.RGBA, tol int) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	w := b.Dx()
	h := b.Dy()
	if w <= 0 || h <= 0 {
		return dst
	}

	pix := src.Pix
	stride := src.Stride
	minX := b.Min.X
	minY := b.Min.Y

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			rowOff := (y - minY) * stride
			off := rowOff + (x-minX)*4
			cr := pix[off+0]
			cg := pix[off+1]
			cb := pix[off+2]
			ca := pix[off+3]
			lc := lumaFromRGB(cr, cg, cb)

			type pt struct{ x, y int }
			neigh := [...]pt{
				{x - 1, y},
				{x + 1, y},
				{x, y - 1},
				{x, y + 1},
				{x - 1, y - 1},
				{x + 1, y - 1},
				{x - 1, y + 1},
				{x + 1, y + 1},
			}

			// Проверяем, есть ли сильный перепад яркости с соседями
			// и одновременно измеряем максимальный перепад.
			edge := false
			maxDl := 0
			for _, p := range neigh {
				if p.x < b.Min.X || p.x >= b.Max.X || p.y < b.Min.Y || p.y >= b.Max.Y {
					continue
				}
				nRowOff := (p.y - minY) * stride
				nOff := nRowOff + (p.x-minX)*4
				nr := pix[nOff+0]
				ng := pix[nOff+1]
				nb := pix[nOff+2]
				ln := lumaFromRGB(nr, ng, nb)
				dl := int(ln - lc)
				if dl < 0 {
					dl = -dl
				}
				if dl > maxDl {
					maxDl = dl
				}
				if dl > tol*2 {
					edge = true
				}
			}

			if !edge {
				// Нет сильного перепада — оставляем как есть.
				dst.SetRGBA(x, y, color.RGBA{cr, cg, cb, ca})
				continue
			}

			// На границе: сила сглаживания зависит от того,
			// насколько сильно отличаются цвета.
			// - если maxDl небольшой (цвета близки) — сглаживаем сильнее
			// - если maxDl большой (резкая граница) — сглаживаем слабее

			// Считаем средний цвет соседей.
			var sumNR, sumNG, sumNB, nCount int
			for _, p := range neigh {
				if p.x < b.Min.X || p.x >= b.Max.X || p.y < b.Min.Y || p.y >= b.Max.Y {
					continue
				}
				nRowOff := (p.y - minY) * stride
				nOff := nRowOff + (p.x-minX)*4
				sumNR += int(pix[nOff+0])
				sumNG += int(pix[nOff+1])
				sumNB += int(pix[nOff+2])
				nCount++
			}

			if nCount == 0 {
				// На краю изображения — просто оставляем исходный.
				dst.SetRGBA(x, y, color.RGBA{cr, cg, cb, ca})
				continue
			}

			avgNR := float64(sumNR) / float64(nCount)
			avgNG := float64(sumNG) / float64(nCount)
			avgNB := float64(sumNB) / float64(nCount)

			// Вычисляем коэффициент сглаживания alpha в [0..1].
			// Базируемся на maxDl:
			// - небольшие перепады -> среднее сглаживание
			// - средние/большие перепады -> более сильное сглаживание, чтобы убрать "зубья"
			var alpha float64
			switch {
			case maxDl < tol*3:
				// слабая граница — лёгкое сглаживание
				alpha = 0.4
			case maxDl < tol*5:
				// нормальная граница — заметное сглаживание
				alpha = 0.3
			default:
				// очень контрастная граница — тоже сглаживаем ощутимо, но не до мыла
				alpha = 0.1
			}

			// Итоговый цвет: смесь исходного и среднего по соседям.
			outR := float64(cr)*(1-alpha) + avgNR*alpha
			outG := float64(cg)*(1-alpha) + avgNG*alpha
			outB := float64(cb)*(1-alpha) + avgNB*alpha

			dst.SetRGBA(x, y, color.RGBA{
				R: uint8(outR + 0.5),
				G: uint8(outG + 0.5),
				B: uint8(outB + 0.5),
				A: ca,
			})
		}
	}

	return dst
}

// smoothJunctions performs gradient-based smoothing at intersections of blocks
// with a given nominal blockSize (usually params.minBlock from the codec).
func smoothJunctions(img *image.RGBA, blockSize int) *image.RGBA {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	if blockSize <= 1 {
		// сетка слишком плотная — фактически нет "квадратов"
		return img
	}

	if w < 2*blockSize || h < 2*blockSize {
		return img
	}

	pix := img.Pix
	stride := img.Stride
	minX := b.Min.X
	minY := b.Min.Y

	maxRadius := blockSize
	if maxRadius > 3 {
		maxRadius = 3
	}
	const junctionLumaSpread int32 = 32
	const lumaSimThreshold int32 = 8

	similar := func(a, b color.RGBA) bool {
		da := lumaFromRGB(a.R, a.G, a.B) - lumaFromRGB(b.R, b.G, b.B)
		if da < 0 {
			da = -da
		}
		return da <= lumaSimThreshold
	}

	lerp8 := func(a, b uint8, t float64) uint8 {
		return uint8(float64(a)*(1-t) + float64(b)*t + 0.5)
	}

	for y := b.Min.Y + blockSize; y < b.Max.Y; y += blockSize {
		for x := b.Min.X + blockSize; x < b.Max.X; x += blockSize {
			off00 := (y-1-minY)*stride + (x-1-minX)*4
			off10 := (y-1-minY)*stride + (x-minX)*4
			off01 := (y-minY)*stride + (x-1-minX)*4
			off11 := (y-minY)*stride + (x-minX)*4

			c00 := color.RGBA{pix[off00+0], pix[off00+1], pix[off00+2], pix[off00+3]}
			c10 := color.RGBA{pix[off10+0], pix[off10+1], pix[off10+2], pix[off10+3]}
			c01 := color.RGBA{pix[off01+0], pix[off01+1], pix[off01+2], pix[off01+3]}
			c11 := color.RGBA{pix[off11+0], pix[off11+1], pix[off11+2], pix[off11+3]}

			l00 := lumaFromRGB(c00.R, c00.G, c00.B)
			l10 := lumaFromRGB(c10.R, c10.G, c10.B)
			l01 := lumaFromRGB(c01.R, c01.G, c01.B)
			l11 := lumaFromRGB(c11.R, c11.G, c11.B)

			minL, maxL := l00, l00
			for _, v := range []int32{l10, l01, l11} {
				if v < minL {
					minL = v
				}
				if v > maxL {
					maxL = v
				}
			}
			if maxL-minL > junctionLumaSpread {
				continue
			}

			L := 0
			for i := 1; i <= maxRadius && x-i >= b.Min.X; i++ {
				upOff := (y-1-minY)*stride + (x-i-minX)*4
				downOff := (y-minY)*stride + (x-i-minX)*4
				up := color.RGBA{pix[upOff+0], pix[upOff+1], pix[upOff+2], pix[upOff+3]}
				down := color.RGBA{pix[downOff+0], pix[downOff+1], pix[downOff+2], pix[downOff+3]}
				if !similar(up, c00) || !similar(down, c01) {
					break
				}
				L = i
			}

			R := 0
			for i := 1; i <= maxRadius && x-1+i < b.Max.X; i++ {
				upOff := (y-1-minY)*stride + (x-1+i-minX)*4
				downOff := (y-minY)*stride + (x-1+i-minX)*4
				up := color.RGBA{pix[upOff+0], pix[upOff+1], pix[upOff+2], pix[upOff+3]}
				down := color.RGBA{pix[downOff+0], pix[downOff+1], pix[downOff+2], pix[downOff+3]}
				if !similar(up, c10) || !similar(down, c11) {
					break
				}
				R = i
			}

			U := 0
			for i := 1; i <= maxRadius && y-i >= b.Min.Y; i++ {
				leftOff := (y-i-minY)*stride + (x-1-minX)*4
				rightOff := (y-i-minY)*stride + (x-minX)*4
				left := color.RGBA{pix[leftOff+0], pix[leftOff+1], pix[leftOff+2], pix[leftOff+3]}
				right := color.RGBA{pix[rightOff+0], pix[rightOff+1], pix[rightOff+2], pix[rightOff+3]}
				if !similar(left, c00) || !similar(right, c10) {
					break
				}
				U = i
			}

			D := 0
			for i := 1; i <= maxRadius && y-1+i < b.Max.Y; i++ {
				leftOff := (y-1+i-minY)*stride + (x-1-minX)*4
				rightOff := (y-1+i-minY)*stride + (x-minX)*4
				left := color.RGBA{pix[leftOff+0], pix[leftOff+1], pix[leftOff+2], pix[leftOff+3]}
				right := color.RGBA{pix[rightOff+0], pix[rightOff+1], pix[rightOff+2], pix[rightOff+3]}
				if !similar(left, c01) || !similar(right, c11) {
					break
				}
				D = i
			}

			if L == 0 || R == 0 || U == 0 || D == 0 {
				continue
			}

			rectMinX := x - L
			rectMaxX := x - 1 + R
			rectMinY := y - U
			rectMaxY := y - 1 + D

			r00, g00, b00, a00 := c00.R, c00.G, c00.B, c00.A
			r10, g10, b10, a10 := c10.R, c10.G, c10.B, c10.A
			r01, g01, b01, a01 := c01.R, c01.G, c01.B, c01.A
			r11, g11, b11, a11 := c11.R, c11.G, c11.B, c11.A

			width := rectMaxX - rectMinX
			height := rectMaxY - rectMinY

			for py := rectMinY; py <= rectMaxY; py++ {
				v := float64(py-rectMinY) / float64(height)
				for px := rectMinX; px <= rectMaxX; px++ {
					u := float64(px-rectMinX) / float64(width)

					rTop := lerp8(r00, r10, u)
					gTop := lerp8(g00, g10, u)
					bTop := lerp8(b00, b10, u)
					aTop := lerp8(a00, a10, u)

					rBot := lerp8(r01, r11, u)
					gBot := lerp8(g01, g11, u)
					bBot := lerp8(b01, b11, u)
					aBot := lerp8(a01, a11, u)

					r := lerp8(rTop, rBot, v)
					g := lerp8(gTop, gBot, v)
					bc := lerp8(bTop, bBot, v)
					a := lerp8(aTop, aBot, v)

					img.SetRGBA(px, py, color.RGBA{r, g, bc, a})
				}
			}
		}
	}
	return img
}

func smoothFlatAreas(src *image.RGBA, blockSize int) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()

	if blockSize <= 1 {
		return src
	}
	if w < 2*blockSize || h < 2*blockSize {
		return src
	}

	dst := image.NewRGBA(b)
	copy(dst.Pix, src.Pix)

	pix := src.Pix
	stride := src.Stride
	minX := b.Min.X
	minY := b.Min.Y

	const boundaryLumaThreshold int32 = 10

	for x := b.Min.X + blockSize; x < b.Max.X; x += blockSize {
		for y := b.Min.Y; y < b.Max.Y; y++ {
			rowOff := (y - minY) * stride
			offL := rowOff + (x-1-minX)*4
			offR := rowOff + (x-minX)*4
			clR, clG, clB, clA := pix[offL+0], pix[offL+1], pix[offL+2], pix[offL+3]
			crR, crG, crB, crA := pix[offR+0], pix[offR+1], pix[offR+2], pix[offR+3]
			d := lumaFromRGB(clR, clG, clB) - lumaFromRGB(crR, crG, crB)
			if d < 0 {
				d = -d
			}
			if d <= boundaryLumaThreshold {
				r := uint8((uint16(clR) + uint16(crR)) / 2)
				g := uint8((uint16(clG) + uint16(crG)) / 2)
				bc := uint8((uint16(clB) + uint16(crB)) / 2)
				a := uint8((uint16(clA) + uint16(crA)) / 2)
				cAvg := color.RGBA{r, g, bc, a}
				dst.SetRGBA(x-1, y, cAvg)
				dst.SetRGBA(x, y, cAvg)
			}
		}
	}

	for y := b.Min.Y + blockSize; y < b.Max.Y; y += blockSize {
		for x := b.Min.X; x < b.Max.X; x++ {
			rowOffT := (y - 1 - minY) * stride
			rowOffB := (y - minY) * stride
			offT := rowOffT + (x-minX)*4
			offB := rowOffB + (x-minX)*4
			ctR, ctG, ctB, ctA := pix[offT+0], pix[offT+1], pix[offT+2], pix[offT+3]
			cbR, cbG, cbB, cbA := pix[offB+0], pix[offB+1], pix[offB+2], pix[offB+3]
			d := lumaFromRGB(ctR, ctG, ctB) - lumaFromRGB(cbR, cbG, cbB)
			if d < 0 {
				d = -d
			}
			if d <= boundaryLumaThreshold {
				r := uint8((uint16(ctR) + uint16(cbR)) / 2)
				g := uint8((uint16(ctG) + uint16(cbG)) / 2)
				bc := uint8((uint16(ctB) + uint16(cbB)) / 2)
				a := uint8((uint16(ctA) + uint16(cbA)) / 2)
				cAvg := color.RGBA{r, g, bc, a}
				dst.SetRGBA(x, y-1, cAvg)
				dst.SetRGBA(x, y, cAvg)
			}
		}
	}
	return dst
}

func smoothBlocks(src *image.RGBA, blockSize int) *image.RGBA {
	return smoothJunctions(smoothFlatAreas(src, blockSize), blockSize)
}

func NewBitWriter(buf *bytes.Buffer) *BitWriter {
	return &BitWriter{buf: buf}
}

// NewBitReaderFromReader читает остаток r в память и создаёт BitReader.
func NewBitReaderFromReader(r *bytes.Reader) *BitReader {
	rest, _ := io.ReadAll(r)
	return &BitReader{data: rest}
}

// NewBitReaderFromBytes создаёт BitReader из уже прочитанных данных.
func NewBitReaderFromBytes(b []byte) *BitReader {
	return &BitReader{data: b}
}

func EncodeZstd(b io.Writer, raw *bytes.Buffer) error {
	enc, err := zstd.NewWriter(b, zstd.WithEncoderConcurrency(runtime.NumCPU()))
	if err != nil {
		return err
	}
	if _, err := enc.Write(raw.Bytes()); err != nil {
		enc.Close()
		return err
	}
	if err := enc.Close(); err != nil {
		return err
	}
	return nil
}

func DecodeZstd(r io.Reader) ([]byte, error) {
	dec, err := zstd.NewReader(r)
	if err != nil {
		return nil, err
	}
	defer dec.Close()

	plain, err := io.ReadAll(dec)
	if err != nil {
		return nil, err
	}

	if len(plain) < 2 {
		return nil, fmt.Errorf("codec2: truncated payload (no palette)")
	}

	return plain, nil
}

// lumaFromRGB returns integer luma (0..255) from raw RGB channels.
func lumaFromRGB(r, g, b uint8) int32 {
	// Rec. 601-type weights.
	return (299*int32(r) + 587*int32(g) + 114*int32(b) + 500) / 1000
}

// RgbToYCoCg теперь использует стандартное YCbCr из стандартной библиотеки.
// Y: 0..255, Co/Cg здесь храним как отклонение Cb/Cr от 128.
func RgbToYCoCg(c color.RGBA) (y, co, cg int32) {
	// Интерпретируем как стандартное YCbCr из стандартной библиотеки.
	// Y: 0..255, Co/Cg здесь храним как отклонение Cb/Cr от 128.
	yc := color.YCbCrModel.Convert(c).(color.YCbCr)
	y = int32(yc.Y)
	co = int32(int8(yc.Cb - 128)) // signed chroma (Cb)
	cg = int32(int8(yc.Cr - 128)) // signed chroma (Cr)
	return
}

func Abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

// packYCoCg now фактически упаковывает цвет в стандартное YCbCr:
//   - Y (0..255)
//   - Cb/Cb (0..255)
//
// Дальше палитра дельта-кодируется по тем же правилам, что и раньше.
func packYCoCg(c color.RGBA) (byte, byte, byte) {
	yc := color.YCbCrModel.Convert(c).(color.YCbCr)
	return yc.Y, yc.Cb, yc.Cr
}

// unpackYCoCg выполняет обратное преобразование из YCbCr в RGBA
// с использованием стандартных преобразований из пакета image/color.
func unpackYCoCg(yb, cb, crb byte) color.RGBA {
	yc := color.YCbCr{Y: yb, Cb: cb, Cr: crb}
	return color.RGBAModel.Convert(yc).(color.RGBA)
}
