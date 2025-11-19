package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"io"
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

	bw := NewBitWriter(b)
	params := paramsForQuality(quality)

	// Кодируем один прямоугольник целиком, дальше рекурсивно.
	if err := encodeRegion(rgba, 0, 0, int(w), int(h), params, bw); err != nil {
		return nil, err
	}
	if err := bw.Flush(); err != nil {
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
	br := NewBitReaderFromReader(r)

	if err := decodeRegion(dst, 0, 0, w, h, params, br); err != nil {
		return nil, err
	}
	return dst, nil
}

// -----------------------------------------------------------------------------
// Params / helpers
// -----------------------------------------------------------------------------

// codec2Params управляет минимальным размером блока и максимально допустимым разбросом яркости.
type codec2Params struct {
	minBlock  int   // минимальный размер стороны блока
	maxSpread int32 // максимально допустимый разброс яркости для "листа"
}

// простая мапа quality -> параметры.
func paramsForQuality(q int) codec2Params {
	// minBlock: от 4 (высокое качество) до 16 (низкое).
	minB := 4 + (100-q)/10 // q=100 -> 4, q=1 -> ~13
	if minB < 4 {
		minB = 4
	}
	if minB > 16 {
		minB = 16
	}

	// maxSpread: от 16 (высокое качество) до 64 (низкое).
	maxSp := int32(16 + (100-q)/2) // q=100 -> ~16, q=1 -> ~65
	if maxSp < 8 {
		maxSp = 8
	}
	if maxSp > 64 {
		maxSp = 64
	}

	return codec2Params{
		minBlock:  minB,
		maxSpread: maxSp,
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

// lumaSpread вычисляет min/max luma в прямоугольнике.
func lumaSpread(img *image.RGBA, x, y, w, h int) int32 {
	b := img.Bounds()
	maxL := int32(0)
	minL := int32(255)

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
	return maxL - minL
}

// -----------------------------------------------------------------------------
// Recursive region encode/decode
// -----------------------------------------------------------------------------

// encodeRegion рекурсивно кодирует прямоугольник (x,y,w,h).
// Структура дерева:
//
//	1 бит: 1 = лист, 0 = внутренний узел.
//	Лист:
//	  3*8 бит: fg RGB
//	  3*8 бит: bg RGB
//	  w*h бит паттерна (1 - fg, 0 - bg).
func encodeRegion(img *image.RGBA, x, y, w, h int, params codec2Params, bw *BitWriter) error {
	// Базовое условие по размеру.
	if w <= params.minBlock && h <= params.minBlock {
		return encodeLeaf(img, x, y, w, h, bw)
	}

	// Проверяем разброс яркости.
	spread := lumaSpread(img, x, y, w, h)
	if spread <= params.maxSpread {
		// Достаточно ровный блок - кодируем как лист.
		return encodeLeaf(img, x, y, w, h, bw)
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
		if err := encodeRegion(img, x, y, w1, h, params, bw); err != nil {
			return err
		}
		if err := encodeRegion(img, x+w1, y, w2, h, params, bw); err != nil {
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
		return encodeLeaf(img, x, y, w, h, bw)
	}

	// Можно делить по высоте — значит узел внутренний.
	if err := bw.WriteBit(false); err != nil { // 0 = внутренний узел
		return err
	}
	if err := encodeRegion(img, x, y, w, h1, params, bw); err != nil {
		return err
	}
	if err := encodeRegion(img, x, y+h1, w, h2, params, bw); err != nil {
		return err
	}

	return nil
}

// encodeLeaf кодирует конкретный блок как двухцветный паттерн.
func encodeLeaf(img *image.RGBA, x, y, w, h int, bw *BitWriter) error {
	// 1) помечаем лист.
	if err := bw.WriteBit(true); err != nil { // 1 = leaf
		return err
	}

	b := img.Bounds()

	// 2) считаем среднюю яркость для порога.
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
		// Ничего нет, закодируем как черный блок bg, fg = bg.
		for i := 0; i < 6; i++ {
			if err := bw.WriteByte(0); err != nil {
				return err
			}
		}
		// Паттерн весь 0.
		for i := 0; i < w*h; i++ {
			if err := bw.WriteBit(false); err != nil {
				return err
			}
		}
		return nil
	}
	thr := sumL / count

	// 3) разделяем на fg/bg и копим средние цвета.
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
		// Все ушли в фон - сделаем fg == bg.
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

	// 4) пишем fg/bg (по 3 байта RGB).
	for _, v := range []uint8{fg.R, fg.G, fg.B, bg.R, bg.G, bg.B} {
		if err := bw.WriteByte(v); err != nil {
			return err
		}
	}

	// 5) пишем паттерн (1 бит на пиксель, построчно).
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
func decodeRegion(dst *image.RGBA, x, y, w, h int, params codec2Params, br *BitReader) error {
	isLeaf, err := br.ReadBit()
	if err != nil {
		return err
	}
	if isLeaf {
		return decodeLeaf(dst, x, y, w, h, br)
	}

	// Внутренний узел - делим так же, как в encodeRegion.
	if w >= h && w/2 >= params.minBlock {
		w1 := w / 2
		w2 := w - w1
		if err := decodeRegion(dst, x, y, w1, h, params, br); err != nil {
			return err
		}
		if err := decodeRegion(dst, x+w1, y, w2, h, params, br); err != nil {
			return err
		}
	} else {
		h1 := h / 2
		h2 := h - h1
		if h1 <= 0 {
			// Должно совпадать с логикой encodeRegion, но на всякий случай
			return decodeLeaf(dst, x, y, w, h, br)
		}
		if err := decodeRegion(dst, x, y, w, h1, params, br); err != nil {
			return err
		}
		if err := decodeRegion(dst, x, y+h1, w, h2, params, br); err != nil {
			return err
		}
	}

	return nil
}

// decodeLeaf зеркален encodeLeaf.
func decodeLeaf(dst *image.RGBA, x, y, w, h int, br *BitReader) error {
	// 1) читаем fg/bg.
	readByte := func() (byte, error) {
		return br.ReadByte()
	}

	var buf [6]byte
	for i := 0; i < 6; i++ {
		b, err := readByte()
		if err != nil {
			return err
		}
		buf[i] = b
	}

	fg := color.RGBA{R: buf[0], G: buf[1], B: buf[2], A: 255}
	bg := color.RGBA{R: buf[3], G: buf[4], B: buf[5], A: 255}

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
