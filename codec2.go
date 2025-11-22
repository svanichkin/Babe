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
	"math"

	"github.com/klauspost/compress/zstd"
)

// Simple experimental recursive block codec.
// API: Encode(img, quality) and Decode(data).

const (
	magicCodec2 = "BABE\n"
)

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

	// 1) Первый проход: собираем fg/bg для всех листьев.
	var leaves []leafColor
	if err := collectRegionColors(rgba, 0, 0, int(w), int(h), params, &leaves); err != nil {
		return nil, err
	}

	// 2) Строим палитру и индексные пары.
	palette, leafIdx, err := buildPalette(leaves, params)
	if err != nil {
		return nil, err
	}

	// 3) Собираем сырой битстрим.
	var raw bytes.Buffer

	// Сначала палитра: количество + RGB.
	if err := binary.Write(&raw, binary.BigEndian, uint16(len(palette))); err != nil {
		return nil, err
	}
	for _, c := range palette {
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

	// 4) Второй проход: кодируем дерево и паттерны с индексами.
	bw := NewBitWriter(&raw)
	leafPos := 0
	if err := encodeRegion(rgba, 0, 0, int(w), int(h), params, bw, leafIdx, &leafPos); err != nil {
		return nil, err
	}
	if err := bw.Flush(); err != nil {
		return nil, err
	}

	// 5) Zstd как было.
	enc, err := zstd.NewWriter(b)
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

	// читаем палитру
	reader := bytes.NewReader(plain)
	var palCount uint16
	if err := binary.Read(reader, binary.BigEndian, &palCount); err != nil {
		return nil, err
	}
	palette := make([]color.RGBA, palCount)
	for i := 0; i < int(palCount); i++ {
		var rgb [3]byte
		if _, err := io.ReadFull(reader, rgb[:]); err != nil {
			return nil, err
		}
		palette[i] = color.RGBA{R: rgb[0], G: rgb[1], B: rgb[2], A: 255}
	}

	// остаток — битстрим дерева и паттернов
	offset := len(plain) - reader.Len()
	if offset < 0 || offset > len(plain) {
		return nil, fmt.Errorf("codec2: invalid palette offset")
	}
	br := NewBitReaderFromBytes(plain[offset:])

	if err := decodeRegion(dst, 0, 0, w, h, params, br, palette); err != nil {
		return nil, err
	}
	return dst, nil
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
	fg, bg color.RGBA
}

type leafIdxPair struct {
	fg, bg uint16
}

func rgbKey(c color.RGBA) uint32 {
	return uint32(c.R)<<16 | uint32(c.G)<<8 | uint32(c.B)
}

// простая мапа quality -> параметры.
// func paramsForQuality(q int) codec2Params {
// 	if q < 1 {
// 		q = 1
// 	}
// 	if q > 100 {
// 		q = 100
// 	}
// 	qi := float64(100 - q) // 0 (high quality) .. 99 (low quality)

// 	// minBlock: от 2 (высокое качество) до 8 (низкое).
// 	minB := 1 + math.Round(qi/20.0) // q=100 -> ~2, q=1 -> ~7
// 	if minB < 1 {
// 		minB = 1
// 	}
// 	if minB > 8 {
// 		minB = 8
// 	}

// 	// colorTol: при высоком качестве палитра более точная (малый допуск),
// 	// при низком качестве можно сливать более далёкие цвета.
// 	colorTol := 4 + int(qi/20.0) // q=100 -> ~2, q=1 -> ~10
// 	if colorTol < 4 {
// 		colorTol = 4
// 	}
// 	if colorTol > 14 {
// 		colorTol = 14
// 	}

// 	// maxGrad: средняя энергия градиента, при которой блок считается "гладким".
// 	// Диапазон маленький, т.к. gradEnergy обычно 0..~15 на естественных картинках.
// 	// При q=100 блоки будут делиться уже при energy > ~3,
// 	// при q=1 делиться будут только при energy > ~11.
// 	maxG := 3 + math.Round(qi/12.0) // q=100 -> ~3, q=1 -> ~11
// 	if maxG < 3 {
// 		maxG = 3
// 	}
// 	if maxG > 16 {
// 		maxG = 16
// 	}

// 	fmt.Printf("minB: %.f, maxG: %.f\n", minB, maxG)

// 	return codec2Params{
// 		minBlock: int(minB),
// 		maxGrad:  int32(maxG),
// 		colorTol: colorTol,
// 	}
// }

func paramsForQuality(q int) codec2Params {
	if q < 1 {
		q = 1
	}
	if q > 100 {
		q = 100
	}

	qi := 100 - q

	params := codec2Params{
		minBlock: 1 + int(float64(qi)*(4.0/100.0)),
		maxGrad:  int32(ExpMap(qi, 99, 0, 150, 1, -4.0)),
		colorTol: ExpMap(qi, 99, 0, 7, 1, -3.0),
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

// gradEnergy вычисляет среднюю энергию градиента яркости в прямоугольнике.
// Чем больше значение, тем больше текстуры/деталей в блоке.
func gradEnergy(img *image.RGBA, x, y, w, h int) int32 {
	b := img.Bounds()
	var sum int64
	var edges int64

	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				continue
			}
			c := img.RGBAAt(px, py)
			l := luma(c)

			// по x
			if px+1 < b.Max.X {
				cr := img.RGBAAt(px+1, py)
				lr := luma(cr)
				dx := lr - l
				if dx < 0 {
					dx = -dx
				}
				sum += int64(dx)
				edges++
			}
			// по y
			if py+1 < b.Max.Y {
				cd := img.RGBAAt(px, py+1)
				ld := luma(cd)
				dy := ld - l
				if dy < 0 {
					dy = -dy
				}
				sum += int64(dy)
				edges++
			}
		}
	}

	if edges == 0 {
		return 0
	}
	// Средняя величина градиента (0..~255).
	return int32(sum / edges)
}

func computeLeafColors(img *image.RGBA, x, y, w, h int) (color.RGBA, color.RGBA) {
	b := img.Bounds()

	var sumL, count int32
	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				continue
			}
			c := img.RGBAAt(px, py)
			sumL += luma(c)
			count++
		}
	}
	if count == 0 {
		black := color.RGBA{0, 0, 0, 255}
		return black, black
	}
	thr := sumL / count

	var fgR, fgG, fgB, bgR, bgG, bgB int64
	var fgCount, bgCount int64

	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				continue
			}
			c := img.RGBAAt(px, py)
			l := luma(c)
			if l >= thr {
				fgR += int64(c.R)
				fgG += int64(c.G)
				fgB += int64(c.B)
				fgCount++
			} else {
				bgR += int64(c.R)
				bgG += int64(c.G)
				bgB += int64(c.B)
				bgCount++
			}
		}
	}

	if fgCount == 0 {
		fgCount = bgCount
		fgR, fgG, fgB = bgR, bgG, bgB
	}
	if bgCount == 0 {
		bgCount = fgCount
		bgR, bgG, bgB = fgR, fgG, fgB
	}

	fg := color.RGBA{
		R: uint8(fgR / fgCount),
		G: uint8(fgG / fgCount),
		B: uint8(fgB / fgCount),
		A: 255,
	}
	bg := color.RGBA{
		R: uint8(bgR / bgCount),
		G: uint8(bgG / bgCount),
		B: uint8(bgB / bgCount),
		A: 255,
	}
	return fg, bg
}

func collectRegionColors(img *image.RGBA, x, y, w, h int, params codec2Params, leaves *[]leafColor) error {
	// условия листа должны совпадать с encodeRegion
	if w <= params.minBlock && h <= params.minBlock {
		fg, bg := computeLeafColors(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{fg: fg, bg: bg})
		return nil
	}

	energy := gradEnergy(img, x, y, w, h)
	if energy == 0 {
		fg, bg := computeLeafColors(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{fg: fg, bg: bg})
		return nil
	}
	if energy <= params.maxGrad && w <= 4*params.minBlock && h <= 4*params.minBlock {
		fg, bg := computeLeafColors(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{fg: fg, bg: bg})
		return nil
	}

	if w >= h && w/2 >= params.minBlock {
		w1 := w / 2
		w2 := w - w1
		if err := collectRegionColors(img, x, y, w1, h, params, leaves); err != nil {
			return err
		}
		if err := collectRegionColors(img, x+w1, y, w2, h, params, leaves); err != nil {
			return err
		}
		return nil
	}

	h1 := h / 2
	h2 := h - h1
	if h1 < params.minBlock {
		fg, bg := computeLeafColors(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{fg: fg, bg: bg})
		return nil
	}

	if err := collectRegionColors(img, x, y, w, h1, params, leaves); err != nil {
		return err
	}
	if err := collectRegionColors(img, x, y+h1, w, h2, params, leaves); err != nil {
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

func buildPalette(leaves []leafColor, params codec2Params) ([]color.RGBA, []leafIdxPair, error) {
	palette := make([]color.RGBA, 0, len(leaves))
	indexPairs := make([]leafIdxPair, len(leaves))
	m := make(map[uint32]uint16)

	for i, lf := range leaves {
		// ---- fg ----
		kfg := rgbKey(lf.fg)
		idxFg, ok := m[kfg]
		if !ok {
			// Пытаемся найти "близкий" цвет в уже собранной палитре.
			found := -1
			for j, pc := range palette {
				if closeColor(lf.fg, pc, params.colorTol) {
					found = j
					break
				}
			}
			if found >= 0 {
				idxFg = uint16(found)
				// Запоминаем и точный ключ -> этот индекс,
				// чтобы следующие абсолютно такие же пиксели не искали линейно.
				m[kfg] = idxFg
			} else {
				// Новый цвет.
				if len(palette) >= 0xFFFF {
					return nil, nil, fmt.Errorf("palette too large")
				}
				idxFg = uint16(len(palette))
				m[kfg] = idxFg
				palette = append(palette, lf.fg)
			}
		}

		// ---- bg ----
		kbg := rgbKey(lf.bg)
		idxBg, ok := m[kbg]
		if !ok {
			found := -1
			for j, pc := range palette {
				if closeColor(lf.bg, pc, params.colorTol) {
					found = j
					break
				}
			}
			if found >= 0 {
				idxBg = uint16(found)
				m[kbg] = idxBg
			} else {
				if len(palette) >= 0xFFFF {
					return nil, nil, fmt.Errorf("palette too large")
				}
				idxBg = uint16(len(palette))
				m[kbg] = idxBg
				palette = append(palette, lf.bg)
			}
		}

		indexPairs[i] = leafIdxPair{fg: idxFg, bg: idxBg}
	}

	return palette, indexPairs, nil
}

// -----------------------------------------------------------------------------
// Recursive region encode/decode
// -----------------------------------------------------------------------------

// encodeRegion рекурсивно кодирует прямоугольник (x,y,w,h).
// Структура дерева:
//
//	1 бит: 1 = лист, 0 = внутренний узел.
//	Лист:
//	  16+16 бит: индексы fg/bg в глобальной палитре (uint16 big-endian каждый)
//	  w*h бит паттерна (1 - fg, 0 - bg).
func encodeRegion(img *image.RGBA, x, y, w, h int, params codec2Params, bw *BitWriter, leafIdx []leafIdxPair, leafPos *int) error {
	// Базовое условие по размеру.
	if w <= params.minBlock && h <= params.minBlock {
		if *leafPos >= len(leafIdx) {
			return fmt.Errorf("encodeRegion: leaf index overflow (minBlock leaf)")
		}
		idx := leafIdx[*leafPos]
		*leafPos++
		return encodeLeaf(img, x, y, w, h, bw, idx)
	}

	// Оцениваем "шероховатость" блока по энергии градиента.
	energy := gradEnergy(img, x, y, w, h)
	if energy == 0 {
		// Идеально ровный блок - сразу лист (даже если он большой).
		if *leafPos >= len(leafIdx) {
			return fmt.Errorf("encodeRegion: leaf index overflow (energy==0)")
		}
		idx := leafIdx[*leafPos]
		*leafPos++
		return encodeLeaf(img, x, y, w, h, bw, idx)
	}
	if energy <= params.maxGrad && w <= 4*params.minBlock && h <= 4*params.minBlock {
		// Блок достаточно гладкий И уже не гигантский — кодируем как лист.
		if *leafPos >= len(leafIdx) {
			return fmt.Errorf("encodeRegion: leaf index overflow (smooth block leaf)")
		}
		idx := leafIdx[*leafPos]
		*leafPos++
		return encodeLeaf(img, x, y, w, h, bw, idx)
	}

	// Здесь блок считается "пёстрым" и мы хотим его делить.
	// Но сначала убеждаемся, что реально можем разделить по какой-то стороне.
	if w >= h && w/2 >= params.minBlock {
		// Реально делим по ширине, значит этот узел точно внутренний.
		if err := bw.WriteBit(false); err != nil { // 0 = внутренний узел
			return err
		}
		w1 := w / 2
		w2 := w - w1
		if err := encodeRegion(img, x, y, w1, h, params, bw, leafIdx, leafPos); err != nil {
			return err
		}
		if err := encodeRegion(img, x+w1, y, w2, h, params, bw, leafIdx, leafPos); err != nil {
			return err
		}
		return nil
	}

	// Пробуем делить по высоте.
	h1 := h / 2
	h2 := h - h1
	if h1 < params.minBlock {
		// Слишком маленький для деления блок — принудительно лист.
		// ВАЖНО: здесь мы НЕ писали бит "0", сразу кодируем лист.
		if *leafPos >= len(leafIdx) {
			return fmt.Errorf("encodeRegion: leaf index overflow (forced leaf)")
		}
		idx := leafIdx[*leafPos]
		*leafPos++
		return encodeLeaf(img, x, y, w, h, bw, idx)
	}

	// Можно делить по высоте — значит узел внутренний.
	if err := bw.WriteBit(false); err != nil { // 0 = внутренний узел
		return err
	}
	if err := encodeRegion(img, x, y, w, h1, params, bw, leafIdx, leafPos); err != nil {
		return err
	}
	if err := encodeRegion(img, x, y+h1, w, h2, params, bw, leafIdx, leafPos); err != nil {
		return err
	}

	return nil
}

// encodeLeaf кодирует конкретный блок как двухцветный паттерн с индексами палитры.
func encodeLeaf(img *image.RGBA, x, y, w, h int, bw *BitWriter, idx leafIdxPair) error {
	// 1) помечаем лист.
	if err := bw.WriteBit(true); err != nil { // 1 = leaf
		return err
	}

	// 2) пишем индексы fg/bg (uint16 big-endian каждый).
	if err := bw.WriteByte(byte(idx.fg >> 8)); err != nil {
		return err
	}
	if err := bw.WriteByte(byte(idx.fg & 0xFF)); err != nil {
		return err
	}
	if err := bw.WriteByte(byte(idx.bg >> 8)); err != nil {
		return err
	}
	if err := bw.WriteByte(byte(idx.bg & 0xFF)); err != nil {
		return err
	}

	// 3) считаем среднюю яркость для порога.
	b := img.Bounds()
	var sumL, count int32
	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				continue
			}
			c := img.RGBAAt(px, py)
			sumL += luma(c)
			count++
		}
	}
	if count == 0 {
		// Паттерн весь 0.
		for i := 0; i < w*h; i++ {
			if err := bw.WriteBit(false); err != nil {
				return err
			}
		}
		return nil
	}
	thr := sumL / count

	// 4) пишем паттерн (1 бит на пиксель, построчно).
	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				// Вне исходного изображения - считаем фоном.
				if err := bw.WriteBit(false); err != nil {
					return err
				}
				continue
			}
			c := img.RGBAAt(px, py)
			l := luma(c)
			if l >= thr {
				if err := bw.WriteBit(true); err != nil {
					return err
				}
			} else {
				if err := bw.WriteBit(false); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// decodeRegion зеркален encodeRegion.
// Читает дерево из bitstream и закрашивает dst.
func decodeRegion(dst *image.RGBA, x, y, w, h int, params codec2Params, br *BitReader, palette []color.RGBA) error {
	isLeaf, err := br.ReadBit()
	if err != nil {
		return err
	}
	if isLeaf {
		return decodeLeaf(dst, x, y, w, h, br, palette)
	}

	// Внутренний узел - делим так же, как в encodeRegion.
	if w >= h && w/2 >= params.minBlock {
		w1 := w / 2
		w2 := w - w1
		if err := decodeRegion(dst, x, y, w1, h, params, br, palette); err != nil {
			return err
		}
		if err := decodeRegion(dst, x+w1, y, w2, h, params, br, palette); err != nil {
			return err
		}
	} else {
		h1 := h / 2
		h2 := h - h1
		if h1 <= 0 {
			// Должно совпадать с логикой encodeRegion, но на всякий случай
			return decodeLeaf(dst, x, y, w, h, br, palette)
		}
		if err := decodeRegion(dst, x, y, w, h1, params, br, palette); err != nil {
			return err
		}
		if err := decodeRegion(dst, x, y+h1, w, h2, params, br, palette); err != nil {
			return err
		}
	}

	return nil
}

// decodeLeaf зеркален encodeLeaf.
func decodeLeaf(dst *image.RGBA, x, y, w, h int, br *BitReader, palette []color.RGBA) error {
	// 1) читаем fg/bg индексы (uint16 big-endian).
	b1, err := br.ReadByte()
	if err != nil {
		return err
	}
	b2, err := br.ReadByte()
	if err != nil {
		return err
	}
	fgIdx := int(uint16(b1)<<8 | uint16(b2))

	b3, err := br.ReadByte()
	if err != nil {
		return err
	}
	b4, err := br.ReadByte()
	if err != nil {
		return err
	}
	bgIdx := int(uint16(b3)<<8 | uint16(b4))

	if fgIdx < 0 || fgIdx >= len(palette) || bgIdx < 0 || bgIdx >= len(palette) {
		return fmt.Errorf("decodeLeaf: palette index out of range")
	}

	fg := palette[fgIdx]
	bg := palette[bgIdx]

	// 2) читаем паттерн и рисуем.
	bounds := dst.Bounds()
	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			bit, err := br.ReadBit()
			if err != nil {
				return err
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

	return nil
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
