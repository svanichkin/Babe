package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"io"
	"runtime"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// Simple experimental recursive block codec.
// API: Encode(img, quality) and Decode(data).

const (
	magicCodec2 = "BAB2"

	// encoderStripes задаёт максимальное количество вертикальных полос,
	// которые кодируются независимо. Можно менять (например, 4, 5, 8, 10),
	// чтобы управлять параллелизмом/гранулярностью.
)

var encoderStripes = runtime.NumCPU()

// Encode encodes the given image with the given quality (1–100).
// Higher quality => smaller allowed luma spread => больше блоков, но лучше детализация.
func Encode(img image.Image, quality int) ([]byte, error) {
	if quality < 1 {
		quality = 1
	}
	if quality > 100 {
		quality = 100
	}

	// Ensure RGBA for fast access.
	rgba := toRGBA(img)
	b := &bytes.Buffer{}

	// Header: magic(4) + width(uint16) + height(uint16) + quality(uint8)
	if _, err := b.Write([]byte(magicCodec2)); err != nil {
		return nil, err
	}
	w := uint16(rgba.Bounds().Dx())
	h := uint16(rgba.Bounds().Dy())

	if err := binary.Write(b, binary.BigEndian, w); err != nil {
		return nil, err
	}
	if err := binary.Write(b, binary.BigEndian, h); err != nil {
		return nil, err
	}
	if err := b.WriteByte(byte(quality)); err != nil {
		return nil, err
	}

	params := paramsForQuality(quality)

	// 1) Первый проход: параллельно собираем цвета для всех листьев и форму дерева по вертикальным полосам.
	leaves, pattern, err := collectRegionColorsStriped(rgba, params)
	if err != nil {
		return nil, err
	}

	// 2) Строим набор локальных палитр и локальные индексы листьев.
	pals, refs, err := buildPalettes(leaves, params)
	if err != nil {
		return nil, err
	}

	// Преобразуем refs в плоский массив локальных индексов (по порядку листьев).
	leafLocal := make([]uint8, len(refs))
	for i, r := range refs {
		leafLocal[i] = r.idx
	}

	// 3) Собираем сырой битстрим.
	var raw bytes.Buffer

	// Сначала пишем все палитры:
	// uint16 numPalettes, затем для каждой:
	// uint8 palSize, затем palSize * RGB.
	if err := binary.Write(&raw, binary.BigEndian, uint16(len(pals))); err != nil {
		return nil, err
	}
	for _, p := range pals {
		if len(p.colors) > 255 {
			return nil, fmt.Errorf("_encode: palette size %d exceeds 255", len(p.colors))
		}
		if err := raw.WriteByte(byte(len(p.colors))); err != nil {
			return nil, err
		}
		for _, c := range p.colors {
			if err := raw.WriteByte(c.R); err != nil {
				return nil, err
			}
			if err := raw.WriteByte(c.G); err != nil {
				return nil, err
			}
			if err := raw.WriteByte(c.B); err != nil {
				return nil, err
			}
		}
	}

	// 4) Второй проход: кодируем дерево и паттерны с индексами по тем же полосам, что и в collectRegionColorsStriped.
	bw := NewBitWriter(&raw)
	leafPos := 0
	patPos := 0

	stripes := splitStripes(int(h), params.minBlock)
	for _, s := range stripes {
		if err := encodeRegion(rgba, 0, s.y, int(w), s.h, params, bw, leafLocal, &leafPos, pattern, &patPos); err != nil {
			return nil, err
		}
	}
	if err := bw.Flush(); err != nil {
		return nil, err
	}

	// 5) Zstd как было.
	enc, err := zstd.NewWriter(b, zstd.WithEncoderConcurrency(runtime.NumCPU()))
	if err != nil {
		return nil, err
	}
	if _, err := enc.Write(raw.Bytes()); err != nil {
		enc.Close()
		return nil, err
	}
	if err := enc.Close(); err != nil {
		return nil, err
	}

	return b.Bytes(), nil
}

// Decode decodes data produced by Encode.
func Decode(data []byte) (image.Image, error) {
	r := bytes.NewReader(data)

	// Read header.
	magic := make([]byte, len(magicCodec2))
	if _, err := r.Read(magic); err != nil {
		return nil, err
	}
	if string(magic) != magicCodec2 {
		return nil, ErrInvalidMagic
	}

	var w16, h16 uint16
	if err := binary.Read(r, binary.BigEndian, &w16); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.BigEndian, &h16); err != nil {
		return nil, err
	}
	qByte, err := r.ReadByte()
	if err != nil {
		return nil, err
	}
	quality := int(qByte)
	if quality < 1 {
		quality = 1
	}
	if quality > 100 {
		quality = 100
	}

	params := paramsForQuality(quality)
	w, h := int(w16), int(h16)

	dst := image.NewRGBA(image.Rect(0, 0, w, h))

	// Оставшиеся данные — zstd-кадр с битстримом.
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

	// читаем набор палитр
	reader := bytes.NewReader(plain)
	var numPalettes uint16
	if err := binary.Read(reader, binary.BigEndian, &numPalettes); err != nil {
		return nil, err
	}
	palettes := make([][]color.RGBA, numPalettes)
	for i := 0; i < int(numPalettes); i++ {
		szByte, err := reader.ReadByte()
		if err != nil {
			return nil, err
		}
		sz := int(szByte)
		if sz < 0 {
			return nil, fmt.Errorf("codec2: negative palette size")
		}
		p := make([]color.RGBA, sz)
		for j := 0; j < sz; j++ {
			var rgb [3]byte
			if _, err := io.ReadFull(reader, rgb[:]); err != nil {
				return nil, err
			}
			p[j] = color.RGBA{R: rgb[0], G: rgb[1], B: rgb[2], A: 255}
		}
		palettes[i] = p
	}

	// остаток — битстрим дерева и паттернов
	offset := len(plain) - reader.Len()
	if offset < 0 || offset > len(plain) {
		return nil, fmt.Errorf("codec2: invalid palette offset")
	}
	br := NewBitReaderFromBytes(plain[offset:])

	// Первый этап: последовательно читаем дерево по тем же вертикальным полосам
	// и собираем список листьев.
	var jobs []leafJob
	stripes := splitStripes(h, params.minBlock)
	leafCounter := 0
	for _, s := range stripes {
		if err := decodeRegionJobs(0, s.y, w, s.h, params, br, palettes, &leafCounter, &jobs); err != nil {
			return nil, err
		}
	}

	// Второй этап: параллельно заливаем блоки цветом.
	paintLeafJobsParallel(dst, palettes, jobs)

	// Лёгкое сглаживание на границах блоков.
	smoothed := smoothEdges(dst, int(params.maxGrad))
	return smoothed, nil
}

// -----------------------------------------------------------------------------
// Params / helpers
// -----------------------------------------------------------------------------

// codec2Params управляет минимальным размером блока и максимально допустимой "шероховатостью" (энергией градиента).
type codec2Params struct {
	minBlock int   // минимальный размер стороны блока
	maxGrad  int32 // максимально допустимая энергия градиента для "листа"
	colorTol int   // допуск по цвету для слияния палитры (в значениях канала)
}

type leafColor struct {
	c color.RGBA
}

type leafIdx uint16

type leafJob struct {
	x, y, w, h int
	pal        int // индекс палитры
	idx        int // локальный индекс цвета внутри палитры
}

// leafRef указывает на цвет в конкретной палитре:
// pal — индекс палитры, idx — индекс цвета внутри этой палитры.
// Пока не используется в боевом коде, подготовка под многопалитровый формат.
type leafRef struct {
	pal uint16
	idx uint8
}

// palette описывает одну локальную палитру (до 256 цветов).
// Сейчас используется только как заготовка для будущего buildPalettes.
type palette struct {
	colors []color.RGBA
}

type stripeInfo struct {
	y, h int
}

// splitStripes разбивает высоту изображения на не более чем encoderStripes вертикальных полос,
// с учётом минимального размера блока. Эти полосы кодируются независимо и последовательно
// в одном битстриме, а декодер повторяет ту же схему.
func splitStripes(totalH, minBlock int) []stripeInfo {
	if totalH <= 0 {
		return []stripeInfo{{y: 0, h: 0}}
	}
	// Максимально допустимое количество полос.
	n := encoderStripes
	if n < 1 {
		n = 1
	}

	// Не имеет смысла делать полос больше, чем количество минимальных блоков по высоте.
	maxByHeight := totalH / minBlock
	if maxByHeight < 1 {
		maxByHeight = 1
	}
	if n > maxByHeight {
		n = maxByHeight
	}

	if n == 1 {
		return []stripeInfo{{y: 0, h: totalH}}
	}

	// Равномерно распределяем высоту по полосам, остаток раздаём сверху вниз.
	base := totalH / n
	rem := totalH % n
	stripes := make([]stripeInfo, 0, n)

	y := 0
	for i := 0; i < n; i++ {
		h := base
		if i < rem {
			h++
		}
		stripes = append(stripes, stripeInfo{y: y, h: h})
		y += h
	}
	return stripes
}

func rgbKey(c color.RGBA) uint32 {
	return uint32(c.R)<<16 | uint32(c.G)<<8 | uint32(c.B)
}

// простая мапа quality -> параметры.
func paramsForQuality(q int) codec2Params {
	if q < 1 {
		q = 1
	}
	if q > 100 {
		q = 100
	}

	qi := 100 - q

	params := codec2Params{
		minBlock: 1,
		maxGrad:  1 + int32(qi),
		colorTol: 1 + int(float64(qi)*(12.0/100.0)),
	}
	fmt.Printf("minBlock=%d, maxGrad=%d, colorTol=%d\n",
		params.minBlock,
		params.maxGrad,
		params.colorTol,
	)
	return params
}

// toRGBA copies any image.Image into an *image.RGBA with bounds starting at (0,0).
func toRGBA(src image.Image) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
	return dst
}

// luma returns integer luma (0..255) for an RGBA pixel.
func luma(c color.RGBA) int32 {
	// Rec. 601-type weights.
	return (299*int32(c.R) + 587*int32(c.G) + 114*int32(c.B) + 500) / 1000
}

// analyzeBlock computes the total, border, and inner energy (luma spread) and average color in a single scan.
func analyzeBlock(img *image.RGBA, x, y, w, h int) (totalEnergy int32, borderEnergy int32, innerEnergy int32, avg color.RGBA) {
	b := img.Bounds()

	var minL int32 = 255
	var maxL int32 = 0

	var minLBorder int32 = 255
	var maxLBorder int32 = 0
	var minLInner int32 = 255
	var maxLInner int32 = 0

	var haveBorder, haveInner bool

	var sumR, sumG, sumB int64
	var count int64

	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				continue
			}
			c := img.RGBAAt(px, py)
			l := luma(c)

			// Общая энергия по блоку.
			if l < minL {
				minL = l
			}
			if l > maxL {
				maxL = l
			}

			// Проверяем, принадлежит ли пиксель границе блока.
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

			sumR += int64(c.R)
			sumG += int64(c.G)
			sumB += int64(c.B)
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

	avg = color.RGBA{
		R: uint8(sumR / count),
		G: uint8(sumG / count),
		B: uint8(sumB / count),
		A: 255,
	}
	return totalEnergy, borderEnergy, innerEnergy, avg
}

// collectRegionColorsStriped запускает collectRegionColors независимо для нескольких вертикальных полос
// и склеивает результат. Каждая полоса использует свою локальную форму дерева (pattern) и цвета листьев.
func collectRegionColorsStriped(img *image.RGBA, params codec2Params) ([]leafColor, []bool, error) {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()

	stripes := splitStripes(h, params.minBlock)
	if len(stripes) == 1 {
		// Простой случай — однотонный проход без параллелизма.
		var leaves []leafColor
		var pattern []bool
		if err := collectRegionColors(img, 0, 0, w, h, params, &leaves, &pattern); err != nil {
			return nil, nil, err
		}
		return leaves, pattern, nil
	}

	type stripeResult struct {
		leaves  []leafColor
		pattern []bool
		err     error
	}

	results := make([]stripeResult, len(stripes))

	var wg sync.WaitGroup
	for i, s := range stripes {
		wg.Add(1)
		go func(i int, s stripeInfo) {
			defer wg.Done()
			var leaves []leafColor
			var pattern []bool
			err := collectRegionColors(img, 0, s.y, w, s.h, params, &leaves, &pattern)
			results[i] = stripeResult{
				leaves:  leaves,
				pattern: pattern,
				err:     err,
			}
		}(i, s)
	}
	wg.Wait()

	// Проверяем ошибки и склеиваем результаты в порядке полос сверху вниз.
	var totalLeaves int
	var totalPattern int
	for _, r := range results {
		if r.err != nil {
			return nil, nil, r.err
		}
		totalLeaves += len(r.leaves)
		totalPattern += len(r.pattern)
	}

	leaves := make([]leafColor, 0, totalLeaves)
	pattern := make([]bool, 0, totalPattern)
	for _, r := range results {
		leaves = append(leaves, r.leaves...)
		pattern = append(pattern, r.pattern...)
	}

	return leaves, pattern, nil
}

func collectRegionColors(img *image.RGBA, x, y, w, h int, params codec2Params, leaves *[]leafColor, pattern *[]bool) error {
	// Логика условий листа должна совпадать с encodeRegion / decodeRegionJobs.
	// Для каждого узла мы должны записать РОВНО ОДИН бит в pattern:
	// true  = лист
	// false = внутренний узел

	// 1) Базовое условие по размеру — сразу лист.
	if w <= params.minBlock && h <= params.minBlock {
		*pattern = append(*pattern, true)
		_, _, _, avg := analyzeBlock(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{c: avg})
		return nil
	}

	// 2) Оцениваем "шероховатость" блока и одновременно считаем средний цвет.
	energy, borderEnergy, innerEnergy, avg := analyzeBlock(img, x, y, w, h)
	if energy == 0 {
		// Идеально ровный блок - лист.
		*pattern = append(*pattern, true)
		*leaves = append(*leaves, leafColor{c: avg})
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
		*leaves = append(*leaves, leafColor{c: avg})
		return nil
	}

	// 3) Здесь блок считается "пёстрым" и мы хотим его делить.
	// Но сначала убеждаемся, что реально можем разделить по какой-то стороне.
	if w >= h && w/2 >= params.minBlock {
		// Внутренний узел: делим по ширине.
		*pattern = append(*pattern, false)

		w1 := w / 2
		w2 := w - w1
		if err := collectRegionColors(img, x, y, w1, h, params, leaves, pattern); err != nil {
			return err
		}
		if err := collectRegionColors(img, x+w1, y, w2, h, params, leaves, pattern); err != nil {
			return err
		}
		return nil
	}

	// Пробуем делить по высоте.
	h1 := h / 2
	h2 := h - h1
	if h1 < params.minBlock {
		// Слишком маленький для деления блок — принудительно лист.
		*pattern = append(*pattern, true)
		_, _, _, avg := analyzeBlock(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{c: avg})
		return nil
	}

	// Внутренний узел: делим по высоте.
	*pattern = append(*pattern, false)
	if err := collectRegionColors(img, x, y, w, h1, params, leaves, pattern); err != nil {
		return err
	}
	if err := collectRegionColors(img, x, y+h1, w, h2, params, leaves, pattern); err != nil {
		return err
	}

	return nil
}

// closeColor returns true if colors a and b are "close enough" (approximate match).
func closeColor(a, b color.RGBA, tol int) bool {
	dr := int(a.R) - int(b.R)
	if dr < 0 {
		dr = -dr
	}
	dg := int(a.G) - int(b.G)
	if dg < 0 {
		dg = -dg
	}
	db := int(a.B) - int(b.B)
	if db < 0 {
		db = -db
	}
	return dr <= tol && dg <= tol && db <= tol
}

// buildPalettes строит набор локальных палитр и ссылок на цвета для каждого листа.
// Идея: каждые 256 листьев начинается новая палитра. Это упрощает логику:
// - максимум 256 листьев на одну палитру => не переполняем uint8 индекс
// - многие палитры будут маленькими, но это не критично для экспериментов.
//
// Пока функция нигде не используется в боевом коде и служит подготовкой
// под многопалитровый формат.
func buildPalettes(leaves []leafColor, params codec2Params) ([]palette, []leafRef, error) {
	if len(leaves) == 0 {
		return nil, nil, nil
	}

	var pals []palette
	pals = append(pals, palette{colors: make([]color.RGBA, 0, 256)})

	refs := make([]leafRef, len(leaves))
	curPalID := 0

	for i, lf := range leaves {
		// Каждые 256 листьев — новая палитра.
		if i > 0 && (i%256) == 0 {
			pals = append(pals, palette{colors: make([]color.RGBA, 0, 256)})
			curPalID++
		}

		p := &pals[curPalID]

		// Пытаемся найти "близкий" цвет в текущей палитре.
		found := -1
		for j, pc := range p.colors {
			if closeColor(lf.c, pc, params.colorTol) {
				found = j
				break
			}
		}
		if found < 0 {
			// Новый цвет. Теоретически размер палитры не должен превышать 256,
			// потому что у нас максимум 256 листьев на одну палитру.
			if len(p.colors) >= 256 {
				return nil, nil, fmt.Errorf("buildPalettes: palette %d exceeded 256 colors", curPalID)
			}
			p.colors = append(p.colors, lf.c)
			found = len(p.colors) - 1
		}

		refs[i] = leafRef{
			pal: uint16(curPalID),
			idx: uint8(found),
		}
	}

	return pals, refs, nil
}

func buildPalette(leaves []leafColor, params codec2Params) ([]color.RGBA, []leafIdx, error) {
	// Мы хотим приблизиться к идее "локальных" палитр по 256 листьев:
	// каждые 256 листьев формируют свою маленькую палитру, но наружу мы по‑прежнему
	// отдаём одну глобальную палитру и глобальные индексы (uint16), чтобы не ломать формат.
	//
	// Для каждой группы из максимум 256 листьев мы собираем локальную палитру:
	// - ищем "близкий" цвет в пределах этой группы (closeColor + colorTol),
	// - при необходимости добавляем новый цвет в ГЛОБАЛЬНУЮ палитру,
	// - храним для локального индекса ссылку на глобальный индекс.
	//
	// В итоге:
	// - разные группы не смешивают цвета между собой (лучше локальность),
	// - глобальная палитра может расти до 65535 цветов (как и раньше),
	// - формат encode/decode остаётся прежним.

	globalPalette := make([]color.RGBA, 0, len(leaves))
	indexPairs := make([]leafIdx, len(leaves))

	const groupSize = 256

	for base := 0; base < len(leaves); base += groupSize {
		end := base + groupSize
		if end > len(leaves) {
			end = len(leaves)
		}

		// Локальная палитра для текущей группы листьев.
		localColors := make([]color.RGBA, 0, end-base)
		localToGlobal := make([]leafIdx, 0, end-base)

		for i := base; i < end; i++ {
			c := leaves[i].c

			// Ищем близкий цвет в локальной палитре этой группы.
			found := -1
			for j, lc := range localColors {
				if closeColor(c, lc, params.colorTol) {
					found = j
					break
				}
			}

			var gidx leafIdx
			if found >= 0 {
				// Используем уже существующий глобальный индекс,
				// сопоставленный с этим локальным цветом.
				gidx = localToGlobal[found]
			} else {
				// Новый цвет: добавляем его в глобальную палитру
				// и связываем с локальной палитрой.
				if len(globalPalette) >= 0xFFFF {
					return nil, nil, fmt.Errorf("palette too large")
				}
				gidx = leafIdx(len(globalPalette))
				globalPalette = append(globalPalette, c)
				localColors = append(localColors, c)
				localToGlobal = append(localToGlobal, gidx)
			}

			indexPairs[i] = gidx
		}
	}

	return globalPalette, indexPairs, nil
}

// -----------------------------------------------------------------------------
// Recursive region encode/decode
// -----------------------------------------------------------------------------

// encodeRegion рекурсивно кодирует прямоугольник (x,y,w,h) по заранее вычисленному дереву (pattern).
// pattern: []bool, где каждый бит указывает, является ли данный узел листом (true) или внутренним (false).
func encodeRegion(img *image.RGBA, x, y, w, h int, params codec2Params, bw *BitWriter, leafLocal []uint8, leafPos *int, pattern []bool, patPos *int) error {
	// На каждом узле читаем заранее принятие решение: лист или внутренний узел.
	if *patPos >= len(pattern) {
		return fmt.Errorf("encodeRegion: pattern overflow")
	}
	isLeaf := pattern[*patPos]
	*patPos++

	if isLeaf {
		// Лист: берём следующий локальный индекс цвета и кодируем лист.
		if *leafPos >= len(leafLocal) {
			return fmt.Errorf("encodeRegion: leaf index overflow")
		}
		local := leafLocal[*leafPos]
		*leafPos++
		return encodeLeaf(bw, local)
	}

	// Внутренний узел: пишем бит 0 и делим блок так же, как в collectRegionColors/decodeRegionJobs.
	if err := bw.WriteBit(false); err != nil { // 0 = внутренний узел
		return err
	}

	if w >= h && w/2 >= params.minBlock {
		// Делим по ширине.
		w1 := w / 2
		w2 := w - w1
		if err := encodeRegion(img, x, y, w1, h, params, bw, leafLocal, leafPos, pattern, patPos); err != nil {
			return err
		}
		if err := encodeRegion(img, x+w1, y, w2, h, params, bw, leafLocal, leafPos, pattern, patPos); err != nil {
			return err
		}
		return nil
	}

	// Иначе делим по высоте; сюда мы не попадём в тех случаях,
	// когда collectRegionColors принудительно сделал лист (h1 < minBlock),
	// потому что pattern для этого узла уже был true.
	h1 := h / 2
	h2 := h - h1
	if h1 < params.minBlock {
		return fmt.Errorf("encodeRegion: inconsistent pattern/geometry (h1 < minBlock for internal node)")
	}

	if err := encodeRegion(img, x, y, w, h1, params, bw, leafLocal, leafPos, pattern, patPos); err != nil {
		return err
	}
	if err := encodeRegion(img, x, y+h1, w, h2, params, bw, leafLocal, leafPos, pattern, patPos); err != nil {
		return err
	}

	return nil
}

func encodeLeaf(bw *BitWriter, local uint8) error {
	// 1) помечаем лист.
	if err := bw.WriteBit(true); err != nil { // 1 = leaf
		return err
	}
	// 2) пишем локальный индекс цвета (0..255) в текущей палитре.
	if err := bw.WriteByte(local); err != nil {
		return err
	}
	// 3) Паттерн полностью убран из формата, поэтому никаких дополнительных бит
	// не пишем. Вся информация о блоке — это его положение/размер в дереве +
	// индекс палитры, который определяется порядковым номером листа.
	return nil
}

// decodeRegionJobs зеркален decodeRegion, но вместо отрисовки листьев
// собирает список "работ" по заливке блоков цветом. Это позволяет
// считать битстрим последовательно, а рисовать блоки — параллельно.
func decodeRegionJobs(x, y, w, h int, params codec2Params, br *BitReader, palettes [][]color.RGBA, leafCounter *int, jobs *[]leafJob) error {
	isLeaf, err := br.ReadBit()
	if err != nil {
		return err
	}
	if isLeaf {
		// Читаем локальный индекс цвета (uint8) и определяем текущую палитру по номеру листа.
		b, err := br.ReadByte()
		if err != nil {
			return err
		}
		palID := *leafCounter / 256
		if palID < 0 || palID >= len(palettes) {
			return fmt.Errorf("decodeRegionJobs: palette id %d out of range", palID)
		}
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
		*leafCounter++
		return nil
	}

	// Внутренний узел - делим так же, как в encodeRegion.
	if w >= h && w/2 >= params.minBlock {
		w1 := w / 2
		w2 := w - w1
		if err := decodeRegionJobs(x, y, w1, h, params, br, palettes, leafCounter, jobs); err != nil {
			return err
		}
		if err := decodeRegionJobs(x+w1, y, w2, h, params, br, palettes, leafCounter, jobs); err != nil {
			return err
		}
	} else {
		h1 := h / 2
		h2 := h - h1
		if h1 <= 0 {
			// Должно совпадать с логикой encodeRegion, но на всякий случай
			// считаем всё как один лист.
			b, err := br.ReadByte()
			if err != nil {
				return err
			}
			palID := *leafCounter / 256
			if palID < 0 || palID >= len(palettes) {
				return fmt.Errorf("decodeRegionJobs: palette id %d out of range (fallback)", palID)
			}
			local := int(b)
			if local < 0 || local >= len(palettes[palID]) {
				return fmt.Errorf("decodeRegionJobs: local color index %d out of range for palette %d (fallback)", local, palID)
			}
			*jobs = append(*jobs, leafJob{
				x:   x,
				y:   y,
				w:   w,
				h:   h,
				pal: palID,
				idx: local,
			})
			*leafCounter++
			return nil
		}
		if err := decodeRegionJobs(x, y, w, h1, params, br, palettes, leafCounter, jobs); err != nil {
			return err
		}
		if err := decodeRegionJobs(x, y+h1, w, h2, params, br, palettes, leafCounter, jobs); err != nil {
			return err
		}
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
				c := palettes[job.pal][job.idx]
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

func NewBitWriter(buf *bytes.Buffer) *BitWriter {
	return &BitWriter{buf: buf}
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

// BitReader читает биты/байты из []byte.
type BitReader struct {
	data []byte
	pos  int   // индекс байта
	acc  byte  // текущий байт
	nbit uint8 // сколько бит уже прочитано из acc (0..8)
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

// -----------------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------------

var ErrInvalidMagic = errors.New("codec2: invalid magic")

func smoothEdges(src *image.RGBA, tol int) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := src.RGBAAt(x, y)
			lc := luma(c)

			type pt struct{ x, y int }
			neigh := []pt{
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
				nc := src.RGBAAt(p.x, p.y)
				ln := luma(nc)
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
				dst.SetRGBA(x, y, c)
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
				nc := src.RGBAAt(p.x, p.y)
				sumNR += int(nc.R)
				sumNG += int(nc.G)
				sumNB += int(nc.B)
				nCount++
			}

			if nCount == 0 {
				// На краю изображения — просто оставляем исходный.
				dst.SetRGBA(x, y, c)
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
			// switch {
			// case maxDl < tol*2:
			// 	// слабая граница — лёгкое сглаживание
			// 	alpha = 0.4
			// case maxDl < tol*4:
			// 	// нормальная граница — заметное сглаживание
			// 	alpha = 0.6
			// default:
			// 	// очень контрастная граница — тоже сглаживаем ощутимо, но не до мыла
			// 	alpha = 0.5
			// }
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
			outR := float64(c.R)*(1-alpha) + avgNR*alpha
			outG := float64(c.G)*(1-alpha) + avgNG*alpha
			outB := float64(c.B)*(1-alpha) + avgNB*alpha

			avg := color.RGBA{
				R: uint8(outR + 0.5),
				G: uint8(outG + 0.5),
				B: uint8(outB + 0.5),
				A: 255,
			}
			dst.SetRGBA(x, y, avg)
		}
	}

	return dst
}
