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

	"github.com/klauspost/compress/zstd"
)

// Simple experimental recursive block codec.
// API: Encode(img, quality) and Decode(data).

const (
	magicCodec2 = "BAB2"
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
	// Лёгкое сглаживание на границах блоков.
	// Берём maxGrad как ориентир порога чувствительности.
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

	q = 100 - q

	return codec2Params{
		minBlock: 1,
		maxGrad:  int32(5 * (q * q)),
		colorTol: 1,
	}
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

// gradEnergy вычисляет разброс яркости (max-min) в прямоугольнике.
// Чем больше значение, тем больше текстуры/деталей в блоке.
func gradEnergy(img *image.RGBA, x, y, w, h int) int32 {
	b := img.Bounds()
	var minL int32 = 255
	var maxL int32 = 0

	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < b.Min.X || py < b.Min.Y || px >= b.Max.X || py >= b.Max.Y {
				continue
			}
			c := img.RGBAAt(px, py)
			l := luma(c)
			if l < minL {
				minL = l
			}
			if l > maxL {
				maxL = l
			}
		}
	}

	if maxL < minL {
		return 0
	}
	return maxL - minL
}

func computeLeafColor(img *image.RGBA, x, y, w, h int) color.RGBA {
	b := img.Bounds()

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
			sumR += int64(c.R)
			sumG += int64(c.G)
			sumB += int64(c.B)
			count++
		}
	}

	if count == 0 {
		return color.RGBA{0, 0, 0, 255}
	}

	return color.RGBA{
		R: uint8(sumR / count),
		G: uint8(sumG / count),
		B: uint8(sumB / count),
		A: 255,
	}
}

func collectRegionColors(img *image.RGBA, x, y, w, h int, params codec2Params, leaves *[]leafColor) error {
	// условия листа должны совпадать с encodeRegion
	if w <= params.minBlock && h <= params.minBlock {
		c := computeLeafColor(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{c: c})
		return nil
	}

	energy := gradEnergy(img, x, y, w, h)
	if energy == 0 {
		c := computeLeafColor(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{c: c})
		return nil
	}
	if energy <= params.maxGrad && w <= 4*params.minBlock && h <= 4*params.minBlock {
		c := computeLeafColor(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{c: c})
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
		c := computeLeafColor(img, x, y, w, h)
		*leaves = append(*leaves, leafColor{c: c})
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

func buildPalette(leaves []leafColor, params codec2Params) ([]color.RGBA, []leafIdx, error) {
	palette := make([]color.RGBA, 0, len(leaves))
	indexPairs := make([]leafIdx, len(leaves))
	m := make(map[uint32]uint16)

	for i, lf := range leaves {
		// Для каждого листа имеем один средний цвет (теперь lf.c).
		k := rgbKey(lf.c)
		idx, ok := m[k]
		if !ok {
			// Пытаемся найти "близкий" цвет в уже собранной палитре.
			found := -1
			for j, pc := range palette {
				if closeColor(lf.c, pc, params.colorTol) {
					found = j
					break
				}
			}
			if found >= 0 {
				idx = uint16(found)
				// Запоминаем и точный ключ -> этот индекс,
				// чтобы следующие абсолютно такие же пиксели не искали линейно.
				m[k] = idx
			} else {
				// Новый цвет.
				if len(palette) >= 0xFFFF {
					return nil, nil, fmt.Errorf("palette too large")
				}
				idx = uint16(len(palette))
				m[k] = idx
				palette = append(palette, lf.c)
			}
		}

		// Один индекс цвета на блок.
		indexPairs[i] = leafIdx(idx)
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
//	  16 бит: индекс цвета в глобальной палитре (uint16 big-endian)
//	Внутренний узел:
//	  рекурсивно кодирует два подрегиона.
func encodeRegion(img *image.RGBA, x, y, w, h int, params codec2Params, bw *BitWriter, leafIdx []leafIdx, leafPos *int) error {
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

func encodeLeaf(img *image.RGBA, x, y, w, h int, bw *BitWriter, idx leafIdx) error {
	// 1) помечаем лист.
	if err := bw.WriteBit(true); err != nil { // 1 = leaf
		return err
	}

	// 2) пишем один индекс цвета.
	v := uint16(idx)
	if err := bw.WriteByte(byte(v >> 8)); err != nil {
		return err
	}
	if err := bw.WriteByte(byte(v & 0xFF)); err != nil {
		return err
	}

	// 3) Паттерн полностью убран из формата, поэтому никаких дополнительных бит
	// не пишем. Вся информация о блоке — это его положение/размер в дереве + индекс палитры.
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

func decodeLeaf(dst *image.RGBA, x, y, w, h int, br *BitReader, palette []color.RGBA) error {
	// 1) читаем индекс цвета (uint16 big-endian).
	b1, err := br.ReadByte()
	if err != nil {
		return err
	}
	b2, err := br.ReadByte()
	if err != nil {
		return err
	}
	idx := int(uint16(b1)<<8 | uint16(b2))

	if idx < 0 || idx >= len(palette) {
		return fmt.Errorf("decodeLeaf: palette index out of range")
	}

	c := palette[idx]
	bounds := dst.Bounds()

	// 2) Заливаем блок одним цветом.
	for yy := 0; yy < h; yy++ {
		for xx := 0; xx < w; xx++ {
			px := x + xx
			py := y + yy
			if px < bounds.Min.X || py < bounds.Min.Y || px >= bounds.Max.X || py >= bounds.Max.Y {
				continue
			}
			dst.SetRGBA(px, py, c)
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
			// - maxDl чуть выше порога -> alpha ~ 0.6 (сильное сглаживание)
			// - maxDl намного больше -> alpha ~ 0.1 (слабое сглаживание)
			var alpha float64
			switch {
			case maxDl < tol*3:
				alpha = 0.6
			case maxDl < tol*5:
				alpha = 0.3
			default:
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
