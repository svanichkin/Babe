// BABE (Bi-Level Adaptive Block Encoding) is a dual-tone block codec for images.
// It operates in YCbCr color space, uses adaptive block sizes (small and macro blocks)
// and per-channel bi-level patterns, plus a light post-process for deblocking
// and gradient smoothing.

package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"io"
	"runtime"
	"sync"

	"github.com/klauspost/compress/zstd"
)

const (
	codec = "BABE\n"
)

// Default block sizes; these are overridden by compression presets via setBlocksForQuality.
var (
	// base size of the small block (in pixels)
	smallBlock = 1
	// size of the macroblock (in pixels)
	macroBlock = 2
)

// current encode quality in [0..100]; used to drive macro/small decisions.
var encQuality int = 70

// channel presence flags for the header; at least Y must be set.
const (
	channelFlagY  = 1 << 0
	channelFlagCb = 1 << 1
	channelFlagCr = 1 << 2
)

// encodeBW toggles grayscale mode; when true, only the Y channel is stored.
var encodeBW bool

// Quality mapping:
// - quality is in [0..100]
// - smallBlock/macroBlock are chosen from a small set of presets
//   to trade off quality vs. compression.

// setBlocksForQuality configures global smallBlock/macroBlock based on a quality in [0..100].
// It uses a small set of discrete presets to keep the behavior stable and predictable.
func setBlocksForQuality(quality int) error {
	// clamp quality to [0..100]
	if quality < 0 {
		quality = 0
	}
	if quality > 100 {
		quality = 100
	}

	// remember quality globally so heuristics (macro vs small) can use it directly
	encQuality = quality

	switch {
	case quality >= 80:
		// highest quality: smallest macro-blocks
		smallBlock = 1
		macroBlock = 2
	case quality >= 60:
		smallBlock = 1
		macroBlock = 3
	case quality >= 40:
		smallBlock = 2
		macroBlock = 4
	case quality >= 20:
		smallBlock = 3
		macroBlock = 6
	default:
		// lowest quality / highest compression
		smallBlock = 4
		macroBlock = 8
	}

	return nil
}

// bitWriter writes bits to a bytes.Buffer (msb-first in each byte).
// It avoids interface-based io.ByteWriter to keep allocations low.
type bitWriter struct {
	buf  *bytes.Buffer
	byte byte
	n    uint8 // number of bits written (0..8)
}

func newBitWriter(buf *bytes.Buffer) bitWriter {
	return bitWriter{buf: buf}
}

// writeBit writes a single bit (msb-first in byte).
func (bw *bitWriter) writeBit(bit bool) {
	bw.byte <<= 1
	if bit {
		bw.byte |= 1
	}
	bw.n++
	if bw.n == 8 {
		_ = bw.buf.WriteByte(bw.byte)
		bw.byte = 0
		bw.n = 0
	}
}

// writeBits writes n bits from bits (msb-first within the provided n bits).
// For example, if n=4 and bits=0b1011, this writes: 1,0,1,1.
func (bw *bitWriter) writeBits(bits uint64, n uint8) {
	for n > 0 {
		free := uint8(8 - bw.n)
		if free == 0 {
			_ = bw.buf.WriteByte(bw.byte)
			bw.byte = 0
			bw.n = 0
			free = 8
		}

		k := free
		if k > n {
			k = n
		}

		shift := n - k
		chunk := uint8((bits >> shift) & ((1 << k) - 1))

		bw.byte = (bw.byte << k) | byte(chunk)
		bw.n += k
		n -= k

		if bw.n == 8 {
			_ = bw.buf.WriteByte(bw.byte)
			bw.byte = 0
			bw.n = 0
		}
	}
}

// flush writes any remaining bits, left-padded with zeros.
func (bw *bitWriter) flush() {
	if bw.n > 0 {
		bw.byte <<= 8 - bw.n
		_ = bw.buf.WriteByte(bw.byte)
		bw.byte = 0
		bw.n = 0
	}
}

// bitReader reads bits from a byte slice (msb-first in each byte).
type bitReader struct {
	data []byte
	idx  int
	bit  uint8 // bit position in current byte (0..7), msb-first
}

func newBitReader(data []byte) bitReader {
	return bitReader{data: data, idx: 0, bit: 0}
}

// readBit returns the next bit, or error if out of data.
func (br *bitReader) readBit() (bool, error) {
	if br.idx >= len(br.data) {
		return false, io.EOF
	}
	b := br.data[br.idx]
	isSet := (b & (1 << (7 - br.bit))) != 0
	br.bit++
	if br.bit == 8 {
		br.bit = 0
		br.idx++
	}
	return isSet, nil
}

// readBitFast is a no-error variant of readBit. The caller must ensure
// there is enough input data remaining.
func (br *bitReader) readBitFast() bool {
	b := br.data[br.idx]
	isSet := (b & (1 << (7 - br.bit))) != 0
	br.bit++
	if br.bit == 8 {
		br.bit = 0
		br.idx++
	}
	return isSet
}

// readBits reads n bits (1..8) and returns them in the low n bits of the result,
// msb-first within the n bits. For example, if the next bits are 1,0,1,1 and n=4,
// this returns 0b1011.
func (br *bitReader) readBits(n uint8) (uint8, error) {
	if n == 0 || n > 8 {
		return 0, fmt.Errorf("readBits: invalid bit count %d", n)
	}
	if br.idx >= len(br.data) {
		return 0, io.EOF
	}

	rem := uint8(8 - br.bit)
	if n <= rem {
		b := br.data[br.idx]
		shift := rem - n
		out := uint8((b >> shift) & byte((1<<n)-1))
		br.bit += n
		if br.bit == 8 {
			br.bit = 0
			br.idx++
		}
		return out, nil
	}

	// Need bits from the next byte as well.
	if br.idx+1 >= len(br.data) {
		return 0, io.EOF
	}

	b0 := br.data[br.idx]
	b1 := br.data[br.idx+1]

	first := uint8(b0 & byte((1<<rem)-1)) // lower "rem" bits
	n2 := n - rem
	second := uint8(b1 >> (8 - n2))

	out := (first << n2) | second
	br.idx++
	br.bit = n2
	return out, nil
}

// readBitsFast is a no-error variant of readBits. The caller must ensure
// there is enough input data remaining, and 1 <= n <= 8.
func (br *bitReader) readBitsFast(n uint8) uint8 {
	rem := uint8(8 - br.bit)
	if n <= rem {
		b := br.data[br.idx]
		shift := rem - n
		out := uint8((b >> shift) & byte((1<<n)-1))
		br.bit += n
		if br.bit == 8 {
			br.bit = 0
			br.idx++
		}
		return out
	}

	// Need bits from the next byte as well.
	b0 := br.data[br.idx]
	b1 := br.data[br.idx+1]

	first := uint8(b0 & byte((1<<rem)-1)) // lower "rem" bits
	n2 := n - rem
	second := uint8(b1 >> (8 - n2))

	out := (first << n2) | second
	br.idx++
	br.bit = n2
	return out
}

// channel IDs for Y, Cb, Cr.
const (
	chY  = 0
	chCb = 1
	chCr = 2
)

// extractYCbCrPlanes converts an image.Image into three planar Y, Cb, Cr slices.
// Each plane has size w*h and is indexed as plane[y*w + x] with 0 <= x < w and 0 <= y < h.
//
// For common concrete types (RGBA/NRGBA) we bypass img.At/RGBA() and read pixels
// directly from the backing Pix slice to reduce allocations and overhead.
func extractYCbCrPlanes(img image.Image) ([]uint8, []uint8, []uint8, int, int) {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()

	yPlane := make([]uint8, w*h)
	cbPlane := make([]uint8, w*h)
	crPlane := make([]uint8, w*h)

	extractYCbCrPlanesInto(img, yPlane, cbPlane, crPlane)
	return yPlane, cbPlane, crPlane, w, h
}

func extractYCbCrPlanesInto(img image.Image, yPlane, cbPlane, crPlane []uint8) {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()

	switch src := img.(type) {
	case *image.RGBA:
		extractYCbCrFromRGBA(src, yPlane, cbPlane, crPlane, w, h)
	case *image.NRGBA:
		extractYCbCrFromNRGBA(src, yPlane, cbPlane, crPlane, w, h)
	case *image.YCbCr:
		extractYCbCrFromYCbCr(src, yPlane, cbPlane, crPlane, w, h)
	case *image.Gray:
		extractYCbCrFromGray(src, yPlane, cbPlane, crPlane, w, h)
	default:
		// Fallback: generic path using img.At. Still parallelised by rows.
		workers := min(runtime.NumCPU(), h)
		if workers < 1 {
			workers = 1
		}

		rowsPerWorker := (h + workers - 1) / workers

		var wg sync.WaitGroup
		for i := 0; i < workers; i++ {
			y0 := i * rowsPerWorker
			if y0 >= h {
				break
			}
			y1 := y0 + rowsPerWorker
			if y1 > h {
				y1 = h
			}

			wg.Add(1)
			go extractYCbCrFromImageStripe(img, b, w, yPlane, cbPlane, crPlane, y0, y1, &wg)
		}
		wg.Wait()
	}
}

func extractYCbCrFromImageStripe(img image.Image, b image.Rectangle, w int, yPlane, cbPlane, crPlane []uint8, yStart, yEnd int, wg *sync.WaitGroup) {
	defer wg.Done()
	for y := yStart; y < yEnd; y++ {
		baseIdx := y * w
		for x := 0; x < w; x++ {
			c := img.At(b.Min.X+x, b.Min.Y+y)
			r16, g16, b16, _ := c.RGBA()
			r8 := uint8(r16 >> 8)
			g8 := uint8(g16 >> 8)
			b8 := uint8(b16 >> 8)
			ycc := rgbToYCbCr(r8, g8, b8)
			idx := baseIdx + x
			yPlane[idx] = uint8(ycc.Y)
			cbPlane[idx] = uint8(ycc.Cb)
			crPlane[idx] = uint8(ycc.Cr)
		}
	}
}

func extractYCbCrPlanesIntoSerial(img image.Image, yPlane, cbPlane, crPlane []uint8) {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()

	switch src := img.(type) {
	case *image.RGBA:
		extractYCbCrFromRGBASequential(src, yPlane, cbPlane, crPlane, w, h)
	case *image.NRGBA:
		extractYCbCrFromNRGBASequential(src, yPlane, cbPlane, crPlane, w, h)
	case *image.YCbCr:
		extractYCbCrFromYCbCrSequential(src, yPlane, cbPlane, crPlane, w, h)
	case *image.Gray:
		extractYCbCrFromGraySequential(src, yPlane, cbPlane, crPlane, w, h)
	default:
		for y := 0; y < h; y++ {
			baseIdx := y * w
			for x := 0; x < w; x++ {
				c := img.At(b.Min.X+x, b.Min.Y+y)
				r16, g16, b16, _ := c.RGBA()
				r8 := uint8(r16 >> 8)
				g8 := uint8(g16 >> 8)
				b8 := uint8(b16 >> 8)
				ycc := rgbToYCbCr(r8, g8, b8)
				idx := baseIdx + x
				yPlane[idx] = uint8(ycc.Y)
				cbPlane[idx] = uint8(ycc.Cb)
				crPlane[idx] = uint8(ycc.Cr)
			}
		}
	}
}

// extractYCbCrFromRGBA converts an *image.RGBA into planar Y, Cb, Cr slices.
// It assumes dst planes are sized to w*h, where w/h come from src.Bounds().Dx/Dy.
func extractYCbCrFromRGBA(src *image.RGBA, yPlane, cbPlane, crPlane []uint8, w, h int) {
	stride := src.Stride
	pix := src.Pix

	workers := runtime.NumCPU()
	if workers > h {
		workers = h
	}
	if workers < 1 {
		workers = 1
	}

	rowsPerWorker := (h + workers - 1) / workers

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		y0 := i * rowsPerWorker
		if y0 >= h {
			break
		}
		y1 := y0 + rowsPerWorker
		if y1 > h {
			y1 = h
		}

		wg.Add(1)
		go extractYCbCrFromRGBAStripe(pix, stride, w, yPlane, cbPlane, crPlane, y0, y1, &wg)
	}
	wg.Wait()
}

func extractYCbCrFromRGBAStripe(pix []byte, stride, w int, yPlane, cbPlane, crPlane []uint8, yStart, yEnd int, wg *sync.WaitGroup) {
	defer wg.Done()
	for y := yStart; y < yEnd; y++ {
		baseIdx := y * w
		pixRow := y * stride
		for x := 0; x < w; x++ {
			p := pixRow + x*4
			r8 := pix[p+0]
			g8 := pix[p+1]
			b8 := pix[p+2]
			idx := baseIdx + x
			rr, gg, bb := int32(r8), int32(g8), int32(b8)
			yPlane[idx] = uint8((77*rr + 150*gg + 29*bb) >> 8)
			cbPlane[idx] = uint8(((-43*rr - 85*gg + 128*bb) >> 8) + 128)
			crPlane[idx] = uint8(((128*rr - 107*gg - 21*bb) >> 8) + 128)
		}
	}
}

func extractYCbCrFromRGBASequential(src *image.RGBA, yPlane, cbPlane, crPlane []uint8, w, h int) {
	stride := src.Stride
	pix := src.Pix

	for y := 0; y < h; y++ {
		baseIdx := y * w
		pixRow := y * stride
		for x := 0; x < w; x++ {
			p := pixRow + x*4
			r8 := pix[p+0]
			g8 := pix[p+1]
			b8 := pix[p+2]
			idx := baseIdx + x
			rr, gg, bb := int32(r8), int32(g8), int32(b8)
			yPlane[idx] = uint8((77*rr + 150*gg + 29*bb) >> 8)
			cbPlane[idx] = uint8(((-43*rr - 85*gg + 128*bb) >> 8) + 128)
			crPlane[idx] = uint8(((128*rr - 107*gg - 21*bb) >> 8) + 128)
		}
	}
}

// extractYCbCrFromNRGBA converts an *image.NRGBA into planar Y, Cb, Cr slices.
// It reads RGB directly from Pix; for primarily opaque images this matches the old At/RGBA path well enough.
func extractYCbCrFromNRGBA(src *image.NRGBA, yPlane, cbPlane, crPlane []uint8, w, h int) {
	stride := src.Stride
	pix := src.Pix

	workers := runtime.NumCPU()
	if workers > h {
		workers = h
	}
	if workers < 1 {
		workers = 1
	}

	rowsPerWorker := (h + workers - 1) / workers

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		y0 := i * rowsPerWorker
		if y0 >= h {
			break
		}
		y1 := y0 + rowsPerWorker
		if y1 > h {
			y1 = h
		}

		wg.Add(1)
		go extractYCbCrFromNRGBAStripe(pix, stride, w, yPlane, cbPlane, crPlane, y0, y1, &wg)
	}
	wg.Wait()
}

func extractYCbCrFromNRGBAStripe(pix []byte, stride, w int, yPlane, cbPlane, crPlane []uint8, yStart, yEnd int, wg *sync.WaitGroup) {
	defer wg.Done()
	for y := yStart; y < yEnd; y++ {
		baseIdx := y * w
		pixRow := y * stride
		for x := 0; x < w; x++ {
			p := pixRow + x*4
			r8 := pix[p+0]
			g8 := pix[p+1]
			b8 := pix[p+2]
			idx := baseIdx + x
			rr, gg, bb := int32(r8), int32(g8), int32(b8)
			yPlane[idx] = uint8((77*rr + 150*gg + 29*bb) >> 8)
			cbPlane[idx] = uint8(((-43*rr - 85*gg + 128*bb) >> 8) + 128)
			crPlane[idx] = uint8(((128*rr - 107*gg - 21*bb) >> 8) + 128)
		}
	}
}

func extractYCbCrFromNRGBASequential(src *image.NRGBA, yPlane, cbPlane, crPlane []uint8, w, h int) {
	stride := src.Stride
	pix := src.Pix

	for y := 0; y < h; y++ {
		baseIdx := y * w
		pixRow := y * stride
		for x := 0; x < w; x++ {
			p := pixRow + x*4
			r8 := pix[p+0]
			g8 := pix[p+1]
			b8 := pix[p+2]
			idx := baseIdx + x
			rr, gg, bb := int32(r8), int32(g8), int32(b8)
			yPlane[idx] = uint8((77*rr + 150*gg + 29*bb) >> 8)
			cbPlane[idx] = uint8(((-43*rr - 85*gg + 128*bb) >> 8) + 128)
			crPlane[idx] = uint8(((128*rr - 107*gg - 21*bb) >> 8) + 128)
		}
	}
}

func extractYCbCrFromYCbCr(src *image.YCbCr, yPlane, cbPlane, crPlane []uint8, w, h int) {
	b := src.Bounds()
	minX, minY := b.Min.X, b.Min.Y

	workers := runtime.NumCPU()
	if workers > h {
		workers = h
	}
	if workers < 1 {
		workers = 1
	}
	rowsPerWorker := (h + workers - 1) / workers

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		y0 := i * rowsPerWorker
		if y0 >= h {
			break
		}
		y1 := y0 + rowsPerWorker
		if y1 > h {
			y1 = h
		}

		wg.Add(1)
		go extractYCbCrFromYCbCrStripe(src, minX, minY, w, yPlane, cbPlane, crPlane, y0, y1, &wg)
	}
	wg.Wait()
}

func extractYCbCrFromYCbCrStripe(src *image.YCbCr, minX, minY, w int, yPlane, cbPlane, crPlane []uint8, yStart, yEnd int, wg *sync.WaitGroup) {
	defer wg.Done()
	rectMinX, rectMinY := src.Rect.Min.X, src.Rect.Min.Y
	yStride := src.YStride
	yPix := src.Y
	cbPix := src.Cb
	crPix := src.Cr

	for y := yStart; y < yEnd; y++ {
		yAbs := minY + y
		baseIdx := y * w
		yRow := (yAbs - rectMinY) * yStride
		for x := 0; x < w; x++ {
			xAbs := minX + x
			yPlane[baseIdx+x] = yPix[yRow+(xAbs-rectMinX)]
			ci := src.COffset(xAbs, yAbs)
			cbPlane[baseIdx+x] = cbPix[ci]
			crPlane[baseIdx+x] = crPix[ci]
		}
	}
}

func extractYCbCrFromYCbCrSequential(src *image.YCbCr, yPlane, cbPlane, crPlane []uint8, w, h int) {
	b := src.Bounds()
	minX, minY := b.Min.X, b.Min.Y
	rectMinX, rectMinY := src.Rect.Min.X, src.Rect.Min.Y

	yStride := src.YStride
	yPix := src.Y
	cbPix := src.Cb
	crPix := src.Cr

	for y := 0; y < h; y++ {
		yAbs := minY + y
		baseIdx := y * w
		yRow := (yAbs - rectMinY) * yStride
		for x := 0; x < w; x++ {
			xAbs := minX + x
			yPlane[baseIdx+x] = yPix[yRow+(xAbs-rectMinX)]
			ci := src.COffset(xAbs, yAbs)
			cbPlane[baseIdx+x] = cbPix[ci]
			crPlane[baseIdx+x] = crPix[ci]
		}
	}
}

func extractYCbCrFromGray(src *image.Gray, yPlane, cbPlane, crPlane []uint8, w, h int) {
	b := src.Bounds()
	minX, minY := b.Min.X, b.Min.Y
	rectMinX, rectMinY := src.Rect.Min.X, src.Rect.Min.Y
	stride := src.Stride
	pix := src.Pix

	workers := runtime.NumCPU()
	if workers > h {
		workers = h
	}
	if workers < 1 {
		workers = 1
	}
	rowsPerWorker := (h + workers - 1) / workers

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		y0 := i * rowsPerWorker
		if y0 >= h {
			break
		}
		y1 := y0 + rowsPerWorker
		if y1 > h {
			y1 = h
		}

		wg.Add(1)
		go extractYCbCrFromGrayStripe(pix, stride, minX, minY, rectMinX, rectMinY, w, yPlane, cbPlane, crPlane, y0, y1, &wg)
	}
	wg.Wait()
}

func extractYCbCrFromGrayStripe(pix []byte, stride, minX, minY, rectMinX, rectMinY, w int, yPlane, cbPlane, crPlane []uint8, yStart, yEnd int, wg *sync.WaitGroup) {
	defer wg.Done()
	for y := yStart; y < yEnd; y++ {
		yAbs := minY + y
		baseIdx := y * w
		pixRow := (yAbs - rectMinY) * stride
		for x := 0; x < w; x++ {
			xAbs := minX + x
			v := pix[pixRow+(xAbs-rectMinX)]
			yPlane[baseIdx+x] = v
			cbPlane[baseIdx+x] = 128
			crPlane[baseIdx+x] = 128
		}
	}
}

func extractYCbCrFromGraySequential(src *image.Gray, yPlane, cbPlane, crPlane []uint8, w, h int) {
	b := src.Bounds()
	minX, minY := b.Min.X, b.Min.Y
	rectMinX, rectMinY := src.Rect.Min.X, src.Rect.Min.Y
	stride := src.Stride
	pix := src.Pix

	for y := 0; y < h; y++ {
		yAbs := minY + y
		baseIdx := y * w
		pixRow := (yAbs - rectMinY) * stride
		for x := 0; x < w; x++ {
			xAbs := minX + x
			v := pix[pixRow+(xAbs-rectMinX)]
			yPlane[baseIdx+x] = v
			cbPlane[baseIdx+x] = 128
			crPlane[baseIdx+x] = 128
		}
	}
}

// canUseBigBlockChannel decides whether a macroBlock region can be encoded as a single block
// for the given channel plane. It uses a quality-dependent spread threshold: lower quality
// allows larger spread (more macroBlocks), higher quality reduces spread (more small blocks).
func allowedMacroSpreadForQuality(quality int) int32 {
	// quality-dependent spread:
	// in each quality band (0–19, 20–39, 40–59, 60–79, 80–100) that we use
	// for smallBlock/macroBlock selection, the allowed spread changes linearly
	// from 64 (bottom of the band) down to 8 (top of the band).
	q := quality
	if q < 0 {
		q = 0
	}
	if q > 100 {
		q = 100
	}

	// Use the same quality bands as setBlocksForQuality:
	//   [0–19]  → 4/8
	//   [20–39] → 3/6
	//   [40–59] → 2/4
	//   [60–79] → 1/3
	//   [80–100]→ 1/2
	var groupStart, groupEnd int
	switch {
	case q < 20:
		groupStart, groupEnd = 0, 19
	case q < 40:
		groupStart, groupEnd = 20, 39
	case q < 60:
		groupStart, groupEnd = 40, 59
	case q < 80:
		groupStart, groupEnd = 60, 79
	default:
		groupStart, groupEnd = 80, 100
	}

	// Linear interpolation of the allowed spread from 64 down to 8 within the band.
	const spreadMax int32 = 64
	const spreadMin int32 = 8
	span := groupEnd - groupStart
	var spread int32
	if span <= 0 {
		spread = (spreadMax + spreadMin) / 2
	} else {
		offset := q - groupStart
		spread = spreadMax - int32(offset)*(spreadMax-spreadMin)/int32(span)
	}

	return spread
}

func canUseBigBlockChannel(plane []uint8, stride, height, x0, y0 int, spread int32) bool {
	if spread <= 0 {
		return false
	}
	if stride <= 0 || height <= 0 || x0 < 0 || y0 < 0 || macroBlock <= 0 {
		return false
	}
	if x0+macroBlock > stride || y0+macroBlock > height {
		return false
	}

	if macroBlock == 2 {
		idx0 := y0*stride + x0
		idx1 := idx0 + stride

		v0 := plane[idx0]
		v1 := plane[idx0+1]
		v2 := plane[idx1]
		v3 := plane[idx1+1]

		minV := v0
		maxV := v0
		if v1 < minV {
			minV = v1
		} else if v1 > maxV {
			maxV = v1
		}
		if v2 < minV {
			minV = v2
		} else if v2 > maxV {
			maxV = v2
		}
		if v3 < minV {
			minV = v3
		} else if v3 > maxV {
			maxV = v3
		}
		return int32(maxV)-int32(minV) < spread
	}

	minV := plane[y0*stride+x0]
	maxV := minV

	for yy := 0; yy < macroBlock; yy++ {
		row := (y0+yy)*stride + x0
		for xx := 0; xx < macroBlock; xx++ {
			v := plane[row+xx]
			if v < minV {
				minV = v
			} else if v > maxV {
				maxV = v
			}
			if int32(maxV)-int32(minV) >= spread {
				return false
			}
		}
	}

	return int32(maxV)-int32(minV) < spread
}

// encodeBlockPlane encodes a single block for one planar channel:
// - computes a mean-based threshold
// - computes FG/BG levels
// - returns whether a bi-level pattern is actually needed (FG != BG)
// - writes pattern bits to pw only when needed
func encodeBlockPlane(plane []uint8, stride, height, x0, y0, bw, bh int, pw *bitWriter) (uint8, uint8, bool, error) {
	total := bw * bh
	if total <= 0 {
		return 0, 0, false, fmt.Errorf("invalid block size")
	}

	if bw == 1 && bh == 1 {
		if x0 < 0 || y0 < 0 || x0 >= stride {
			return 0, 0, false, fmt.Errorf("encodeBlockPlane: index out of range")
		}
		idx := y0*stride + x0
		if idx < 0 || idx >= len(plane) {
			return 0, 0, false, fmt.Errorf("encodeBlockPlane: index out of range")
		}
		v := plane[idx]
		return v, v, false, nil
	}

	if x0 < 0 || y0 < 0 || bw <= 0 || bh <= 0 || stride <= 0 || height <= 0 || x0+bw > stride {
		return 0, 0, false, fmt.Errorf("encodeBlockPlane: index out of range")
	}
	if y0+bh > height {
		return 0, 0, false, fmt.Errorf("encodeBlockPlane: index out of range")
	}

	if bw == 2 && bh == 2 {
		idx0 := y0*stride + x0
		idx1 := idx0 + stride

		v0 := plane[idx0]
		v1 := plane[idx0+1]
		v2 := plane[idx1]
		v3 := plane[idx1+1]

		sum := uint64(v0) + uint64(v1) + uint64(v2) + uint64(v3)
		avg := uint8(sum / 4)
		thr := avg

		var fgSum, bgSum uint64
		var fgCnt, bgCnt uint32

		var bits uint64
		isFg0 := v0 >= thr
		bits <<= 1
		if isFg0 {
			bits |= 1
			fgSum += uint64(v0)
			fgCnt++
		} else {
			bgSum += uint64(v0)
			bgCnt++
		}

		isFg1 := v1 >= thr
		bits <<= 1
		if isFg1 {
			bits |= 1
			fgSum += uint64(v1)
			fgCnt++
		} else {
			bgSum += uint64(v1)
			bgCnt++
		}

		isFg2 := v2 >= thr
		bits <<= 1
		if isFg2 {
			bits |= 1
			fgSum += uint64(v2)
			fgCnt++
		} else {
			bgSum += uint64(v2)
			bgCnt++
		}

		isFg3 := v3 >= thr
		bits <<= 1
		if isFg3 {
			bits |= 1
			fgSum += uint64(v3)
			fgCnt++
		} else {
			bgSum += uint64(v3)
			bgCnt++
		}

		if fgCnt == 0 || bgCnt == 0 {
			return avg, avg, false, nil
		}
		fg := uint8(fgSum / uint64(fgCnt))
		bg := uint8(bgSum / uint64(bgCnt))
		if fg == bg {
			return fg, bg, false, nil
		}
		if pw != nil {
			pw.writeBits(bits, 4)
		}
		return fg, bg, true, nil
	}

	// Read the block once into a small stack buffer.
	var valsBuf [64]uint8
	if total > len(valsBuf) {
		return 0, 0, false, fmt.Errorf("encodeBlockPlane: block too large")
	}
	vals := valsBuf[:total]

	var sum uint64
	i := 0
	for yy := 0; yy < bh; yy++ {
		row := (y0+yy)*stride + x0
		for xx := 0; xx < bw; xx++ {
			v := plane[row+xx]
			vals[i] = v
			sum += uint64(v)
			i++
		}
	}
	thr := uint8(sum / uint64(total))

	var fgSum, bgSum uint64
	var fgCnt, bgCnt uint32

	var bits uint64
	for _, v := range vals {
		isFg := v >= thr
		bits <<= 1
		if isFg {
			bits |= 1
			fgSum += uint64(v)
			fgCnt++
		} else {
			bgSum += uint64(v)
			bgCnt++
		}
	}

	avg := uint8(sum / uint64(total))
	if fgCnt == 0 || bgCnt == 0 {
		return avg, avg, false, nil
	}
	fg := uint8(fgSum / uint64(fgCnt))
	bg := uint8(bgSum / uint64(bgCnt))
	if fg == bg {
		return fg, bg, false, nil
	}
	if pw != nil {
		pw.writeBits(bits, uint8(total))
	}
	return fg, bg, true, nil
}

// encodeChannel builds one complete stream for a single planar channel:
// its own block count, size stream, pattern stream, and per-block FG/BG levels.
func encodeChannel(plane []uint8, stride, w4, h4, fullW, fullH int, useMacro bool) (uint32, []byte, []byte, []byte, []uint8, []uint8, error) {
	// macro-block decision bits (only for main fullW x fullH area)
	// Precompute an upper bound on the total block count so we can
	// preallocate FG/BG slices and avoid repeated growth.
	macroGroupsX := 0
	macroGroupsY := 0
	if macroBlock > 0 {
		macroGroupsX = fullW / macroBlock
		macroGroupsY = fullH / macroBlock
	}
	macroGroupCount := macroGroupsX * macroGroupsY

	// worst-case: every macro-block is split into a full grid of small blocks
	mbs := 1
	if smallBlock > 0 {
		mbs = macroBlock / smallBlock
		if mbs < 1 {
			mbs = 1
		}
	}
	worstMainBlocks := macroGroupCount * mbs * mbs

	// right stripe: only small blocks
	rightWidth := w4 - fullW
	rightBlocks := 0
	if rightWidth > 0 && smallBlock > 0 {
		rightBlocks = (fullH / smallBlock) * (rightWidth / smallBlock)
	}

	// bottom stripe (including bottom-right corner): only small blocks
	bottomHeight := h4 - fullH
	bottomBlocks := 0
	if bottomHeight > 0 && smallBlock > 0 {
		bottomBlocks = (bottomHeight / smallBlock) * (w4 / smallBlock)
	}

	worstBlockCount := worstMainBlocks + rightBlocks + bottomBlocks
	if worstBlockCount < 0 {
		worstBlockCount = 0
	}

	var blockCount uint32

	var sizeBuf bytes.Buffer
	if macroGroupCount > 0 {
		sizeBuf.Grow((macroGroupCount + 7) / 8)
	}
	sizeW := newBitWriter(&sizeBuf)
	var patternBuf bytes.Buffer
	// At most one pattern bit per pixel in the encoded area.
	// For smallBlock==1, patterns are only emitted for macro-blocks, so fullW*fullH is a safe bound.
	patternBits := w4 * h4
	if smallBlock == 1 {
		patternBits = fullW * fullH
	}
	if patternBits > 0 {
		patternBuf.Grow((patternBits + 7) / 8)
	}
	patternW := newBitWriter(&patternBuf)

	var typeBuf bytes.Buffer
	if worstBlockCount > 0 {
		typeBuf.Grow((worstBlockCount + 7) / 8)
	}
	typeW := newBitWriter(&typeBuf)

	var fgVals []uint8
	var bgVals []uint8
	if worstBlockCount > 0 {
		fgVals = make([]uint8, 0, worstBlockCount)
		bgVals = make([]uint8, 0, worstBlockCount)
	}

	height := h4
	spread := allowedMacroSpreadForQuality(encQuality)

	// main macroBlock x macroBlock area
	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			useBig := useMacro && canUseBigBlockChannel(plane, stride, height, mx, my, spread)
			sizeW.writeBit(useBig)
			if useBig {
				fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, mx, my, macroBlock, macroBlock, &patternW)
				if err != nil {
					return 0, nil, nil, nil, nil, nil, err
				}
				fgVals = append(fgVals, fg)
				if isPattern {
					bgVals = append(bgVals, bg)
				}
				typeW.writeBit(isPattern)
				blockCount++
			} else {
				// grid of smallBlock x smallBlock blocks covering the macroBlock area
				for by := 0; by < macroBlock; by += smallBlock {
					for bx := 0; bx < macroBlock; bx += smallBlock {
						xx, yy := mx+bx, my+by
						// for smallBlock > 1 we also emit pattern bits; for smallBlock == 1 we skip pattern bits
						var pw *bitWriter
						if smallBlock > 1 {
							pw = &patternW
						}
						fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, xx, yy, smallBlock, smallBlock, pw)
						if err != nil {
							return 0, nil, nil, nil, nil, nil, err
						}
						fgVals = append(fgVals, fg)
						if isPattern {
							bgVals = append(bgVals, bg)
						}
						typeW.writeBit(isPattern)
						blockCount++
					}
				}
			}
		}
	}

	// right stripe: small blocks only
	for my := 0; my < fullH; my += smallBlock {
		for mx := fullW; mx < w4; mx += smallBlock {
			var pw *bitWriter
			if smallBlock > 1 {
				pw = &patternW
			}
			fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			fgVals = append(fgVals, fg)
			if isPattern {
				bgVals = append(bgVals, bg)
			}
			typeW.writeBit(isPattern)
			blockCount++
		}
	}
	// bottom stripe: small blocks only (including bottom-right corner)
	for my := fullH; my < h4; my += smallBlock {
		for mx := 0; mx < w4; mx += smallBlock {
			var pw *bitWriter
			if smallBlock > 1 {
				pw = &patternW
			}
			fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			fgVals = append(fgVals, fg)
			if isPattern {
				bgVals = append(bgVals, bg)
			}
			typeW.writeBit(isPattern)
			blockCount++
		}
	}

	sizeW.flush()
	typeW.flush()
	patternW.flush()

	return blockCount, sizeBuf.Bytes(), typeBuf.Bytes(), patternBuf.Bytes(), fgVals, bgVals, nil
}

type encoderChannelScratch struct {
	sizeBuf    bytes.Buffer
	typeBuf    bytes.Buffer
	patternBuf bytes.Buffer
	fgVals     []uint8
	bgVals     []uint8
}

type encodeChannelSpec struct {
	id    int
	plane []uint8
}

type encodeChannelResult struct {
	blockCount   uint32
	sizeBytes    []byte
	typeBytes    []byte
	patternBytes []byte
	fgVals       []uint8
	bgVals       []uint8
	err          error
}

func encodeChannelWorker(e *Encoder, dst *encodeChannelResult, plane []uint8, stride, w4, h4, fullW, fullH int, useMacro bool, scratch *encoderChannelScratch, wg *sync.WaitGroup) {
	defer wg.Done()
	blockCount, sizeBytes, typeBytes, patternBytes, fgVals, bgVals, err := e.encodeChannelReuse(plane, stride, w4, h4, fullW, fullH, useMacro, scratch)
	dst.blockCount = blockCount
	dst.sizeBytes = sizeBytes
	dst.typeBytes = typeBytes
	dst.patternBytes = patternBytes
	dst.fgVals = fgVals
	dst.bgVals = bgVals
	dst.err = err
}

// Encoder reuses large scratch buffers across Encode calls to reduce allocations.
// It is not safe for concurrent use. The returned []byte is reused and will be
// overwritten on the next Encode call.
type Encoder struct {
	// Parallel enables internal goroutines (plane extraction and per-channel encode).
	// Set to false to reduce goroutine overhead and allocations.
	Parallel bool

	yPlane  []uint8
	cbPlane []uint8
	crPlane []uint8

	raw  bytes.Buffer
	bw   *bufio.Writer
	comp []byte

	ch [3]encoderChannelScratch

	zenc *zstd.Encoder
}

func NewEncoder() *Encoder {
	e := &Encoder{}
	e.Parallel = true
	e.bw = bufio.NewWriter(&e.raw)
	e.zenc = mustNewZstdEncoder()
	return e
}

func (e *Encoder) ensurePlanes(w, h int) {
	n := w * h
	if cap(e.yPlane) < n {
		e.yPlane = make([]uint8, n)
		e.cbPlane = make([]uint8, n)
		e.crPlane = make([]uint8, n)
		return
	}
	e.yPlane = e.yPlane[:n]
	e.cbPlane = e.cbPlane[:n]
	e.crPlane = e.crPlane[:n]
}

func (e *Encoder) encodeChannelReuse(plane []uint8, stride, w4, h4, fullW, fullH int, useMacro bool, scratch *encoderChannelScratch) (uint32, []byte, []byte, []byte, []uint8, []uint8, error) {
	// macro-block decision bits (only for main fullW x fullH area)
	// Precompute an upper bound on the total block count so we can
	// preallocate FG/BG slices and avoid repeated growth.
	macroGroupsX := 0
	macroGroupsY := 0
	if macroBlock > 0 {
		macroGroupsX = fullW / macroBlock
		macroGroupsY = fullH / macroBlock
	}
	macroGroupCount := macroGroupsX * macroGroupsY

	// worst-case: every macro-block is split into a full grid of small blocks
	mbs := 1
	if smallBlock > 0 {
		mbs = macroBlock / smallBlock
		if mbs < 1 {
			mbs = 1
		}
	}
	worstMainBlocks := macroGroupCount * mbs * mbs

	// right stripe: only small blocks
	rightWidth := w4 - fullW
	rightBlocks := 0
	if rightWidth > 0 && smallBlock > 0 {
		rightBlocks = (fullH / smallBlock) * (rightWidth / smallBlock)
	}

	// bottom stripe (including bottom-right corner): only small blocks
	bottomHeight := h4 - fullH
	bottomBlocks := 0
	if bottomHeight > 0 && smallBlock > 0 {
		bottomBlocks = (bottomHeight / smallBlock) * (w4 / smallBlock)
	}

	worstBlockCount := worstMainBlocks + rightBlocks + bottomBlocks
	if worstBlockCount < 0 {
		worstBlockCount = 0
	}

	scratch.sizeBuf.Reset()
	scratch.typeBuf.Reset()
	scratch.patternBuf.Reset()

	if macroGroupCount > 0 {
		scratch.sizeBuf.Grow((macroGroupCount + 7) / 8)
	}
	if worstBlockCount > 0 {
		scratch.typeBuf.Grow((worstBlockCount + 7) / 8)
	}
	// At most one pattern bit per pixel in the encoded area.
	// For smallBlock==1, patterns are only emitted for macro-blocks, so fullW*fullH is a safe bound.
	patternBits := w4 * h4
	if smallBlock == 1 {
		patternBits = fullW * fullH
	}
	if patternBits > 0 {
		scratch.patternBuf.Grow((patternBits + 7) / 8)
	}

	sizeW := newBitWriter(&scratch.sizeBuf)
	typeW := newBitWriter(&scratch.typeBuf)
	patternW := newBitWriter(&scratch.patternBuf)

	if worstBlockCount > 0 {
		if cap(scratch.fgVals) < worstBlockCount {
			scratch.fgVals = make([]uint8, 0, worstBlockCount)
			scratch.bgVals = make([]uint8, 0, worstBlockCount)
		} else {
			scratch.fgVals = scratch.fgVals[:0]
			scratch.bgVals = scratch.bgVals[:0]
		}
	} else {
		scratch.fgVals = scratch.fgVals[:0]
		scratch.bgVals = scratch.bgVals[:0]
	}

	var blockCount uint32
	height := h4
	spread := allowedMacroSpreadForQuality(encQuality)

	if useMacro && smallBlock == 1 && macroBlock == 2 {
		// Specialized hot path for the most common setting (quality >= 80):
		// - macro blocks are 2x2
		// - small blocks are 1x1 (always solid, no pattern bits)
		// This avoids canUseBigBlockChannel/encodeBlockPlane calls for the 1x1 case
		// and encodes 2x2 blocks with direct 4-byte loads.
		for my := 0; my < fullH; my += 2 {
			row0 := my * stride
			row1 := (my + 1) * stride
			for mx := 0; mx < fullW; mx += 2 {
				idx0 := row0 + mx
				idx2 := row1 + mx

				v0 := plane[idx0]
				v1 := plane[idx0+1]
				v2 := plane[idx2]
				v3 := plane[idx2+1]

				minV := v0
				maxV := v0
				if v1 < minV {
					minV = v1
				} else if v1 > maxV {
					maxV = v1
				}
				if v2 < minV {
					minV = v2
				} else if v2 > maxV {
					maxV = v2
				}
				if v3 < minV {
					minV = v3
				} else if v3 > maxV {
					maxV = v3
				}

				useBig := int32(maxV)-int32(minV) < spread
				sizeW.writeBit(useBig)

				if useBig {
					sum := uint64(v0) + uint64(v1) + uint64(v2) + uint64(v3)
					avg := uint8(sum / 4)
					thr := avg

					var bits uint64
					var fgSum, bgSum uint64
					var fgCnt, bgCnt uint32

					isFg0 := v0 >= thr
					bits <<= 1
					if isFg0 {
						bits |= 1
						fgSum += uint64(v0)
						fgCnt++
					} else {
						bgSum += uint64(v0)
						bgCnt++
					}

					isFg1 := v1 >= thr
					bits <<= 1
					if isFg1 {
						bits |= 1
						fgSum += uint64(v1)
						fgCnt++
					} else {
						bgSum += uint64(v1)
						bgCnt++
					}

					isFg2 := v2 >= thr
					bits <<= 1
					if isFg2 {
						bits |= 1
						fgSum += uint64(v2)
						fgCnt++
					} else {
						bgSum += uint64(v2)
						bgCnt++
					}

					isFg3 := v3 >= thr
					bits <<= 1
					if isFg3 {
						bits |= 1
						fgSum += uint64(v3)
						fgCnt++
					} else {
						bgSum += uint64(v3)
						bgCnt++
					}

					fg, bg := avg, avg
					isPattern := false
					if fgCnt != 0 && bgCnt != 0 {
						fg = uint8(fgSum / uint64(fgCnt))
						bg = uint8(bgSum / uint64(bgCnt))
						isPattern = fg != bg
					}

					scratch.fgVals = append(scratch.fgVals, fg)
					if isPattern {
						patternW.writeBits(bits, 4)
						scratch.bgVals = append(scratch.bgVals, bg)
					}
					typeW.writeBit(isPattern)
					blockCount++
				} else {
					// 4 solid 1x1 blocks, no pattern bits, type bits are all 0.
					scratch.fgVals = append(scratch.fgVals, v0, v1, v2, v3)
					typeW.writeBits(0, 4)
					blockCount += 4
				}
			}
		}

		// right stripe: 1x1 solid blocks only
		for my := 0; my < fullH; my++ {
			for mx := fullW; mx < w4; mx++ {
				v := plane[my*stride+mx]
				scratch.fgVals = append(scratch.fgVals, v)
				typeW.writeBit(false)
				blockCount++
			}
		}

		// bottom stripe: 1x1 solid blocks only (including bottom-right corner)
		for my := fullH; my < h4; my++ {
			row := my * stride
			for mx := 0; mx < w4; mx++ {
				v := plane[row+mx]
				scratch.fgVals = append(scratch.fgVals, v)
				typeW.writeBit(false)
				blockCount++
			}
		}

		sizeW.flush()
		typeW.flush()
		patternW.flush()

		return blockCount, scratch.sizeBuf.Bytes(), scratch.typeBuf.Bytes(), scratch.patternBuf.Bytes(), scratch.fgVals, scratch.bgVals, nil
	}

	// main macroBlock x macroBlock area
	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			useBig := useMacro && canUseBigBlockChannel(plane, stride, height, mx, my, spread)
			sizeW.writeBit(useBig)
			if useBig {
				fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, mx, my, macroBlock, macroBlock, &patternW)
				if err != nil {
					return 0, nil, nil, nil, nil, nil, err
				}
				scratch.fgVals = append(scratch.fgVals, fg)
				if isPattern {
					scratch.bgVals = append(scratch.bgVals, bg)
				}
				typeW.writeBit(isPattern)
				blockCount++
			} else {
				// grid of smallBlock x smallBlock blocks covering the macroBlock area
				for by := 0; by < macroBlock; by += smallBlock {
					for bx := 0; bx < macroBlock; bx += smallBlock {
						xx, yy := mx+bx, my+by
						// for smallBlock > 1 we also emit pattern bits; for smallBlock == 1 we skip pattern bits
						var pw *bitWriter
						if smallBlock > 1 {
							pw = &patternW
						}
						fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, xx, yy, smallBlock, smallBlock, pw)
						if err != nil {
							return 0, nil, nil, nil, nil, nil, err
						}
						scratch.fgVals = append(scratch.fgVals, fg)
						if isPattern {
							scratch.bgVals = append(scratch.bgVals, bg)
						}
						typeW.writeBit(isPattern)
						blockCount++
					}
				}
			}
		}
	}

	// right stripe: small blocks only
	for my := 0; my < fullH; my += smallBlock {
		for mx := fullW; mx < w4; mx += smallBlock {
			var pw *bitWriter
			if smallBlock > 1 {
				pw = &patternW
			}
			fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			scratch.fgVals = append(scratch.fgVals, fg)
			if isPattern {
				scratch.bgVals = append(scratch.bgVals, bg)
			}
			typeW.writeBit(isPattern)
			blockCount++
		}
	}
	// bottom stripe: small blocks only (including bottom-right corner)
	for my := fullH; my < h4; my += smallBlock {
		for mx := 0; mx < w4; mx += smallBlock {
			var pw *bitWriter
			if smallBlock > 1 {
				pw = &patternW
			}
			fg, bg, isPattern, err := encodeBlockPlane(plane, stride, height, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			scratch.fgVals = append(scratch.fgVals, fg)
			if isPattern {
				scratch.bgVals = append(scratch.bgVals, bg)
			}
			typeW.writeBit(isPattern)
			blockCount++
		}
	}

	sizeW.flush()
	typeW.flush()
	patternW.flush()

	return blockCount, scratch.sizeBuf.Bytes(), scratch.typeBuf.Bytes(), scratch.patternBuf.Bytes(), scratch.fgVals, scratch.bgVals, nil
}

func (e *Encoder) Encode(img image.Image, quality int, bwmode bool) ([]byte, error) {
	encodeBW = bwmode

	if err := setBlocksForQuality(quality); err != nil {
		return nil, err
	}
	if e.zenc == nil {
		e.zenc = mustNewZstdEncoder()
	}

	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()

	e.ensurePlanes(w, h)
	if e.Parallel {
		extractYCbCrPlanesInto(img, e.yPlane, e.cbPlane, e.crPlane)
	} else {
		extractYCbCrPlanesIntoSerial(img, e.yPlane, e.cbPlane, e.crPlane)
	}

	// Decide which channels will be stored. Y is always present; Cb/Cr
	// may be omitted in grayscale mode.
	channelsMask := byte(channelFlagY)
	if !encodeBW {
		channelsMask |= channelFlagCb | channelFlagCr
	}

	e.raw.Reset()
	e.bw.Reset(&e.raw)

	w4 := (w / smallBlock) * smallBlock
	h4 := (h / smallBlock) * smallBlock
	if w4 == 0 || h4 == 0 {
		return nil, fmt.Errorf("image too small for %dx%d blocks: %dx%d", smallBlock, smallBlock, w, h)
	}
	fullW := (w4 / macroBlock) * macroBlock
	fullH := (h4 / macroBlock) * macroBlock

	if macroBlock < smallBlock || macroBlock%smallBlock != 0 {
		return nil, fmt.Errorf("macroBlock (%d) must be >= smallBlock (%d) and a multiple of it",
			macroBlock, smallBlock)
	}
	useMacro := macroBlock > smallBlock

	// --- Write header ---
	if _, err := e.bw.WriteString(codec); err != nil {
		return nil, err
	}
	if err := writeU16BE(e.bw, uint16(smallBlock)); err != nil {
		return nil, err
	}
	if err := writeU16BE(e.bw, uint16(macroBlock)); err != nil {
		return nil, err
	}
	// channels mask: which Y/Cb/Cr planes are stored.
	if err := e.bw.WriteByte(channelsMask); err != nil {
		return nil, err
	}
	if err := writeU32BE(e.bw, uint32(w)); err != nil {
		return nil, err
	}
	if err := writeU32BE(e.bw, uint32(h)); err != nil {
		return nil, err
	}

	var channels [3]encodeChannelSpec
	chCount := 1
	channels[0] = encodeChannelSpec{id: chY, plane: e.yPlane}
	if !encodeBW {
		channels[1] = encodeChannelSpec{id: chCb, plane: e.cbPlane}
		channels[2] = encodeChannelSpec{id: chCr, plane: e.crPlane}
		chCount = 3
	}

	if e.Parallel {
		// Encode channels in parallel; scratch is per-channel so it's safe to reuse.
		var results [3]encodeChannelResult
		var wg sync.WaitGroup
		for i := 0; i < chCount; i++ {
			wg.Add(1)
			ch := channels[i]
			go encodeChannelWorker(e, &results[i], ch.plane, w, w4, h4, fullW, fullH, useMacro, &e.ch[ch.id], &wg)
		}
		wg.Wait()

		for i := 0; i < chCount; i++ {
			res := results[i]
			if res.err != nil {
				return nil, res.err
			}

			// number of blocks for this channel
			if err := writeU32BE(e.bw, res.blockCount); err != nil {
				return nil, err
			}
			// write size stream for this channel
			if err := writeU32BE(e.bw, uint32(len(res.sizeBytes))); err != nil {
				return nil, err
			}
			if _, err := e.bw.Write(res.sizeBytes); err != nil {
				return nil, err
			}
			// write type stream for this channel
			if err := writeU32BE(e.bw, uint32(len(res.typeBytes))); err != nil {
				return nil, err
			}
			if _, err := e.bw.Write(res.typeBytes); err != nil {
				return nil, err
			}
			// write pattern stream for this channel
			if err := writeU32BE(e.bw, uint32(len(res.patternBytes))); err != nil {
				return nil, err
			}
			if _, err := e.bw.Write(res.patternBytes); err != nil {
				return nil, err
			}

			if err := writeU32BE(e.bw, uint32(len(res.fgVals))); err != nil {
				return nil, err
			}
			if err := writeDeltaPackedBytes(e.bw, res.fgVals); err != nil {
				return nil, err
			}

			if err := writeU32BE(e.bw, uint32(len(res.bgVals))); err != nil {
				return nil, err
			}
			if err := writeDeltaPackedBytes(e.bw, res.bgVals); err != nil {
				return nil, err
			}
		}
	} else {
		// Encode channels sequentially (no additional goroutines).
		for i := 0; i < chCount; i++ {
			ch := channels[i]
			scratch := &e.ch[ch.id]
			blockCount, sizeBytes, typeBytes, patternBytes, fgVals, bgVals, err := e.encodeChannelReuse(ch.plane, w, w4, h4, fullW, fullH, useMacro, scratch)
			if err != nil {
				return nil, err
			}

			if err := writeU32BE(e.bw, blockCount); err != nil {
				return nil, err
			}
			if err := writeU32BE(e.bw, uint32(len(sizeBytes))); err != nil {
				return nil, err
			}
			if _, err := e.bw.Write(sizeBytes); err != nil {
				return nil, err
			}
			if err := writeU32BE(e.bw, uint32(len(typeBytes))); err != nil {
				return nil, err
			}
			if _, err := e.bw.Write(typeBytes); err != nil {
				return nil, err
			}
			if err := writeU32BE(e.bw, uint32(len(patternBytes))); err != nil {
				return nil, err
			}
			if _, err := e.bw.Write(patternBytes); err != nil {
				return nil, err
			}

			if err := writeU32BE(e.bw, uint32(len(fgVals))); err != nil {
				return nil, err
			}
			if err := writeDeltaPackedBytes(e.bw, fgVals); err != nil {
				return nil, err
			}

			if err := writeU32BE(e.bw, uint32(len(bgVals))); err != nil {
				return nil, err
			}
			if err := writeDeltaPackedBytes(e.bw, bgVals); err != nil {
				return nil, err
			}
		}
	}

	if err := e.bw.Flush(); err != nil {
		return nil, err
	}

	e.comp = e.zenc.EncodeAll(e.raw.Bytes(), e.comp[:0])
	return e.comp, nil
}

// EncodeTo encodes the image and writes the compressed result to w.
// This mirrors the common Go codec style (e.g. image/jpeg), while still reusing
// Encoder internal scratch buffers between calls.
func (e *Encoder) EncodeTo(w io.Writer, img image.Image, quality int, bwmode bool) error {
	comp, err := e.Encode(img, quality, bwmode)
	if err != nil {
		return err
	}
	_, err = w.Write(comp)
	return err
}

// Encode runs the BABE encoder with three fully independent channel streams (Y, Cb, Cr).
// Each channel has its own block list, size stream, pattern stream, and FG/BG levels.
func Encode(img image.Image, quality int, bwmode bool) ([]byte, error) {
	encodeBW = bwmode

	if err := setBlocksForQuality(quality); err != nil {
		return nil, err
	}

	yPlane, cbPlane, crPlane, w, h := extractYCbCrPlanes(img)

	// Decide which channels will be stored. Y is always present; Cb/Cr
	// may be omitted in grayscale mode.
	channelsMask := byte(channelFlagY)
	if !encodeBW {
		channelsMask |= channelFlagCb | channelFlagCr
	}

	var raw bytes.Buffer
	bw := bufio.NewWriter(&raw)

	w4 := (w / smallBlock) * smallBlock
	h4 := (h / smallBlock) * smallBlock
	if w4 == 0 || h4 == 0 {
		return nil, fmt.Errorf("image too small for %dx%d blocks: %dx%d", smallBlock, smallBlock, w, h)
	}
	fullW := (w4 / macroBlock) * macroBlock
	fullH := (h4 / macroBlock) * macroBlock

	if macroBlock < smallBlock || macroBlock%smallBlock != 0 {
		return nil, fmt.Errorf("macroBlock (%d) must be >= smallBlock (%d) and a multiple of it",
			macroBlock, smallBlock)
	}
	useMacro := macroBlock > smallBlock

	// --- Write header ---
	if _, err := bw.WriteString(codec); err != nil {
		return nil, err
	}
	if err := writeU16BE(bw, uint16(smallBlock)); err != nil {
		return nil, err
	}
	if err := writeU16BE(bw, uint16(macroBlock)); err != nil {
		return nil, err
	}
	// channels mask: which Y/Cb/Cr planes are stored.
	if err := bw.WriteByte(channelsMask); err != nil {
		return nil, err
	}
	if err := writeU32BE(bw, uint32(w)); err != nil {
		return nil, err
	}
	if err := writeU32BE(bw, uint32(h)); err != nil {
		return nil, err
	}

	// --- Encode channel(s) depending on grayscale mode ---
	type channelResult struct {
		blockCount   uint32
		sizeBytes    []byte
		typeBytes    []byte
		patternBytes []byte
		fgVals       []uint8
		bgVals       []uint8
		err          error
	}

	var wg sync.WaitGroup

	type channelSpec struct {
		id    int
		plane []uint8
	}

	var channels []channelSpec
	// Y is always present.
	channels = append(channels, channelSpec{id: chY, plane: yPlane})
	// Cb/Cr are stored only if not in grayscale mode.
	if !encodeBW {
		channels = append(channels, channelSpec{id: chCb, plane: cbPlane})
		channels = append(channels, channelSpec{id: chCr, plane: crPlane})
	}

	results := make([]channelResult, len(channels))

	for i := range channels {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ch := channels[i]
			blockCount, sizeBytes, typeBytes, patternBytes, fgVals, bgVals, err := encodeChannel(ch.plane, w, w4, h4, fullW, fullH, useMacro)
			results[i] = channelResult{
				blockCount:   blockCount,
				sizeBytes:    sizeBytes,
				typeBytes:    typeBytes,
				patternBytes: patternBytes,
				fgVals:       fgVals,
				bgVals:       bgVals,
				err:          err,
			}
		}(i)
	}

	wg.Wait()

	// Write channels in fixed order of IDs: always Y first, then optional Cb/Cr.
	// Solid blocks store only FG; BG is implicit == FG. Pattern blocks store both FG and BG.
	for i := range channels {
		res := results[i]
		if res.err != nil {
			return nil, res.err
		}
		blockCount := res.blockCount
		// number of blocks for this channel
		if err := writeU32BE(bw, blockCount); err != nil {
			return nil, err
		}
		// write size stream for this channel
		if err := writeU32BE(bw, uint32(len(res.sizeBytes))); err != nil {
			return nil, err
		}
		if _, err := bw.Write(res.sizeBytes); err != nil {
			return nil, err
		}
		// write type stream for this channel
		if err := writeU32BE(bw, uint32(len(res.typeBytes))); err != nil {
			return nil, err
		}
		if _, err := bw.Write(res.typeBytes); err != nil {
			return nil, err
		}
		// write pattern stream for this channel
		if err := writeU32BE(bw, uint32(len(res.patternBytes))); err != nil {
			return nil, err
		}
		if _, err := bw.Write(res.patternBytes); err != nil {
			return nil, err
		}
		fgLen := uint32(len(res.fgVals))
		if err := writeU32BE(bw, fgLen); err != nil {
			return nil, err
		}
		if err := writeDeltaPackedBytes(bw, res.fgVals); err != nil {
			return nil, err
		}

		bgLen := uint32(len(res.bgVals))
		if err := writeU32BE(bw, bgLen); err != nil {
			return nil, err
		}
		if err := writeDeltaPackedBytes(bw, res.bgVals); err != nil {
			return nil, err
		}
	}

	if err := bw.Flush(); err != nil {
		return nil, err
	}
	rawBytes := raw.Bytes()
	comp, err := compressZstd(rawBytes)
	if err != nil {
		return nil, fmt.Errorf("zstd encode: %w", err)
	}

	return comp, nil
}

// drawBlockPlane decodes a single block for one channel into a planar buffer.
func drawBlockPlane(plane []uint8, stride int, x0, y0, bw, bh int, br *bitReader, fg, bg uint8) error {
	for yy := 0; yy < bh; yy++ {
		for xx := 0; xx < bw; xx++ {
			bit, err := br.readBit()
			if err != nil {
				return err
			}
			val := bg
			if bit {
				val = fg
			}
			idx := (y0+yy)*stride + (x0 + xx)
			if idx >= len(plane) {
				return fmt.Errorf("drawBlockPlane: index out of range")
			}
			plane[idx] = val
		}
	}
	return nil
}

// fillBlockPlane fills a block with a single value (no pattern bits).
func fillBlockPlane(plane []uint8, stride int, x0, y0, bw, bh int, val uint8) error {
	for yy := 0; yy < bh; yy++ {
		for xx := 0; xx < bw; xx++ {
			idx := (y0+yy)*stride + (x0 + xx)
			if idx >= len(plane) {
				return fmt.Errorf("fillBlockPlane: index out of range")
			}
			plane[idx] = val
		}
	}
	return nil
}

func drawBlockPix(pix []byte, strideBytes int, x0, y0, bw, bh int, br *bitReader, fg, bg uint8, channelOffset int) error {
	if bw == 1 && bh == 1 {
		bit, err := br.readBit()
		if err != nil {
			return err
		}
		o := y0*strideBytes + x0*4 + channelOffset
		if o < 0 || o >= len(pix) {
			return fmt.Errorf("drawBlockPix: index out of range")
		}
		if bit {
			pix[o] = fg
		} else {
			pix[o] = bg
		}
		return nil
	}

	if bw == 2 && bh == 2 {
		bits, err := br.readBits(4)
		if err != nil {
			return err
		}

		row0 := y0*strideBytes + x0*4 + channelOffset
		row1 := row0 + strideBytes
		o00 := row0
		o01 := row0 + 4
		o10 := row1
		o11 := row1 + 4

		if o00 < 0 || o11 >= len(pix) {
			return fmt.Errorf("drawBlockPix: index out of range")
		}

		if (bits & 0b1000) != 0 {
			pix[o00] = fg
		} else {
			pix[o00] = bg
		}
		if (bits & 0b0100) != 0 {
			pix[o01] = fg
		} else {
			pix[o01] = bg
		}
		if (bits & 0b0010) != 0 {
			pix[o10] = fg
		} else {
			pix[o10] = bg
		}
		if (bits & 0b0001) != 0 {
			pix[o11] = fg
		} else {
			pix[o11] = bg
		}
		return nil
	}

	for yy := 0; yy < bh; yy++ {
		row := (y0+yy)*strideBytes + x0*4 + channelOffset
		for xx := 0; xx < bw; xx++ {
			bit, err := br.readBit()
			if err != nil {
				return err
			}
			val := bg
			if bit {
				val = fg
			}
			o := row + xx*4
			if o < 0 || o >= len(pix) {
				return fmt.Errorf("drawBlockPix: index out of range")
			}
			pix[o] = val
		}
	}
	return nil
}

func fillBlockPix(pix []byte, strideBytes int, x0, y0, bw, bh int, val uint8, channelOffset int) error {
	if bw == 1 && bh == 1 {
		o := y0*strideBytes + x0*4 + channelOffset
		if o < 0 || o >= len(pix) {
			return fmt.Errorf("fillBlockPix: index out of range")
		}
		pix[o] = val
		return nil
	}

	if bw == 2 && bh == 2 {
		row0 := y0*strideBytes + x0*4 + channelOffset
		row1 := row0 + strideBytes
		o00 := row0
		o01 := row0 + 4
		o10 := row1
		o11 := row1 + 4

		if o00 < 0 || o11 >= len(pix) {
			return fmt.Errorf("fillBlockPix: index out of range")
		}
		pix[o00] = val
		pix[o01] = val
		pix[o10] = val
		pix[o11] = val
		return nil
	}

	for yy := 0; yy < bh; yy++ {
		row := (y0+yy)*strideBytes + x0*4 + channelOffset
		for xx := 0; xx < bw; xx++ {
			o := row + xx*4
			if o < 0 || o >= len(pix) {
				return fmt.Errorf("fillBlockPix: index out of range")
			}
			pix[o] = val
		}
	}
	return nil
}

// readChannelSegment returns a zero-copy slice of the next channel stream
// (as written by Encode) from the decompressed payload. It advances *pos
// to the byte immediately after this segment.
func readChannelSegment(data []byte, pos *int) ([]byte, error) {
	start := *pos

	readU32 := func(label string) (uint32, error) {
		if len(data)-*pos < 4 {
			return 0, fmt.Errorf("readChannelSegment: truncated while reading %s", label)
		}
		v := binary.BigEndian.Uint32(data[*pos : *pos+4])
		*pos += 4
		return v, nil
	}

	// blockCount (4 bytes) – we don't need the value here, just skip it.
	if _, err := readU32("blockCount"); err != nil {
		return nil, err
	}

	// sizeStreamLen (4 bytes) + payload
	sizeStreamLen, err := readU32("sizeStreamLen")
	if err != nil {
		return nil, err
	}
	if uint32(len(data)-*pos) < sizeStreamLen {
		return nil, fmt.Errorf("readChannelSegment: truncated size stream")
	}
	*pos += int(sizeStreamLen)

	// typeStreamLen (4 bytes) + payload
	typeStreamLen, err := readU32("typeStreamLen")
	if err != nil {
		return nil, err
	}
	if uint32(len(data)-*pos) < typeStreamLen {
		return nil, fmt.Errorf("readChannelSegment: truncated type stream")
	}
	*pos += int(typeStreamLen)

	// patternLen (4 bytes) + payload
	patternLen, err := readU32("patternLen")
	if err != nil {
		return nil, err
	}
	if uint32(len(data)-*pos) < patternLen {
		return nil, fmt.Errorf("readChannelSegment: truncated pattern stream")
	}
	*pos += int(patternLen)

	// FG packed length (4 bytes) + payload
	fgPackedLen, err := readU32("FG packedLen")
	if err != nil {
		return nil, err
	}
	if uint32(len(data)-*pos) < fgPackedLen {
		return nil, fmt.Errorf("readChannelSegment: truncated FG packed data")
	}
	*pos += int(fgPackedLen)

	// BG packed length (4 bytes) + payload
	bgPackedLen, err := readU32("BG packedLen")
	if err != nil {
		return nil, err
	}
	if uint32(len(data)-*pos) < bgPackedLen {
		return nil, fmt.Errorf("readChannelSegment: truncated BG packed data")
	}
	*pos += int(bgPackedLen)

	end := *pos
	if end < start || end > len(data) {
		return nil, fmt.Errorf("readChannelSegment: invalid segment bounds")
	}

	return data[start:end], nil
}

// decodeChannel decodes one channel stream into a planar buffer of size imgW x imgH.
func decodeChannel(data []byte, imgW, imgH int) ([]uint8, error) {
	pos := 0

	// helpers to read from the raw byte slice without extra allocations
	readU32 := func(label string) (uint32, error) {
		if len(data)-pos < 4 {
			return 0, fmt.Errorf("decodeChannel: truncated while reading %s", label)
		}
		v := binary.BigEndian.Uint32(data[pos : pos+4])
		pos += 4
		return v, nil
	}

	readSlice := func(n uint32, label string) ([]byte, error) {
		if n > uint32(len(data)-pos) {
			return nil, fmt.Errorf("decodeChannel: truncated while reading %s", label)
		}
		end := pos + int(n)
		s := data[pos:end]
		pos = end
		return s, nil
	}

	// blockCount
	blockCount, err := readU32("blockCount")
	if err != nil {
		return nil, err
	}

	// size stream (bit flags for macro vs small in the main area)
	sizeStreamLen, err := readU32("sizeStreamLen")
	if err != nil {
		return nil, err
	}
	sizeBytes, err := readSlice(sizeStreamLen, "sizeStream")
	if err != nil {
		return nil, err
	}
	sizeBR := newBitReader(sizeBytes)

	// type stream (pattern vs solid blocks)
	typeStreamLen, err := readU32("typeStreamLen")
	if err != nil {
		return nil, err
	}
	typeBytes, err := readSlice(typeStreamLen, "typeStream")
	if err != nil {
		return nil, err
	}
	typeBR := newBitReader(typeBytes)

	// pattern stream (actual bi-level patterns)
	patternLen, err := readU32("patternLen")
	if err != nil {
		return nil, err
	}
	patternBytes, err := readSlice(patternLen, "patternStream")
	if err != nil {
		return nil, err
	}
	patternBR := newBitReader(patternBytes)

	// FG: read mode and decode as a delta stream
	fgPackedLen, err := readU32("FG packedLen")
	if err != nil {
		return nil, err
	}
	fgPacked, err := readSlice(fgPackedLen, "FG packed data")
	if err != nil {
		return nil, err
	}
	if blockCount > 0 && fgPackedLen < blockCount {
		return nil, fmt.Errorf("decodeChannel: FG packed data too short")
	}
	fgStream, err := newDeltaStream(fgPacked, int(blockCount))
	if err != nil {
		return nil, err
	}

	// BG: read mode and decode as a delta stream
	bgPackedLen, err := readU32("BG packedLen")
	if err != nil {
		return nil, err
	}
	bgPacked, err := readSlice(bgPackedLen, "BG packed data")
	if err != nil {
		return nil, err
	}
	if bgPackedLen > blockCount {
		return nil, fmt.Errorf("decodeChannel: BG packed data too long")
	}
	bgStream, err := newDeltaStream(bgPacked, int(bgPackedLen))
	if err != nil {
		return nil, err
	}
	bgPerPattern := true

	// reconstruct the block geometry and fill the plane
	plane := make([]uint8, imgW*imgH)

	w4 := (imgW / smallBlock) * smallBlock
	h4 := (imgH / smallBlock) * smallBlock
	fullW := (w4 / macroBlock) * macroBlock
	fullH := (h4 / macroBlock) * macroBlock
	macroGroupCount := (fullW / macroBlock) * (fullH / macroBlock)
	useMacro := macroBlock > smallBlock

	// rebuild macro-block size decisions from the size stream
	macroBig := make([]bool, macroGroupCount)
	for i := range macroGroupCount {
		bit, err := sizeBR.readBit()
		if err != nil {
			return nil, err
		}
		macroBig[i] = bit
	}

	blockIndex := 0
	macroIndex := 0

	// main macroBlock x macroBlock area
	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			if blockIndex >= int(blockCount) {
				return nil, fmt.Errorf("unexpected end of blocks in main area")
			}
			if useMacro && macroBig[macroIndex] {
				// macroBlock: read one type bit, FG and BG from the delta streams, use full pattern
				bitType, err := typeBR.readBit()
				if err != nil {
					return nil, err
				}

				fg, err := fgStream.next()
				if err != nil {
					return nil, err
				}
				if bitType {
					bg := fg
					if bgPerPattern {
						bg, err = bgStream.next()
						if err != nil {
							return nil, err
						}
					} else {
						bg, err = bgStream.next()
						if err != nil {
							return nil, err
						}
					}
					if err := drawBlockPlane(plane, imgW, mx, my, macroBlock, macroBlock, &patternBR, fg, bg); err != nil {
						return nil, err
					}
				} else {
					if !bgPerPattern {
						// legacy mode stores BG per block; consume to stay in sync
						if _, err := bgStream.next(); err != nil {
							return nil, err
						}
					}
					if err := fillBlockPlane(plane, imgW, mx, my, macroBlock, macroBlock, fg); err != nil {
						return nil, err
					}
				}
				blockIndex++
			} else {
				// grid of smallBlock x smallBlock blocks covering the macroBlock area
				for by := 0; by < macroBlock; by += smallBlock {
					for bx := 0; bx < macroBlock; bx += smallBlock {
						if blockIndex >= int(blockCount) {
							return nil, fmt.Errorf("unexpected end of blocks in macro grid")
						}
						bitType, err := typeBR.readBit()
						if err != nil {
							return nil, err
						}
						fg, err := fgStream.next()
						if err != nil {
							return nil, err
						}
						if bitType {
							bg := fg
							if bgPerPattern {
								bg, err = bgStream.next()
								if err != nil {
									return nil, err
								}
							} else {
								bg, err = bgStream.next()
								if err != nil {
									return nil, err
								}
							}
							if err := drawBlockPlane(plane, imgW, mx+bx, my+by, smallBlock, smallBlock, &patternBR, fg, bg); err != nil {
								return nil, err
							}
						} else {
							if !bgPerPattern {
								// legacy mode stores BG per block; consume to stay in sync
								if _, err := bgStream.next(); err != nil {
									return nil, err
								}
							}
							if err := fillBlockPlane(plane, imgW, mx+bx, my+by, smallBlock, smallBlock, fg); err != nil {
								return nil, err
							}
						}
						blockIndex++
					}
				}
			}
			macroIndex++
		}
	}

	// right stripe: small blocks only
	for my := 0; my < fullH; my += smallBlock {
		for mx := fullW; mx < w4; mx += smallBlock {
			if blockIndex >= int(blockCount) {
				return nil, fmt.Errorf("unexpected end of blocks in right stripe")
			}
			bitType, err := typeBR.readBit()
			if err != nil {
				return nil, err
			}
			fg, err := fgStream.next()
			if err != nil {
				return nil, err
			}
			if bitType {
				bg := fg
				if bgPerPattern {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				} else {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				}
				if err := drawBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, &patternBR, fg, bg); err != nil {
					return nil, err
				}
			} else {
				if !bgPerPattern {
					// legacy mode stores BG per block; consume to stay in sync
					if _, err := bgStream.next(); err != nil {
						return nil, err
					}
				}
				if err := fillBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, fg); err != nil {
					return nil, err
				}
			}
			blockIndex++
		}
	}
	// bottom stripe: small blocks only (including bottom-right corner)
	for my := fullH; my < h4; my += smallBlock {
		for mx := 0; mx < w4; mx += smallBlock {
			if blockIndex >= int(blockCount) {
				return nil, fmt.Errorf("unexpected end of blocks in bottom stripe")
			}
			bitType, err := typeBR.readBit()
			if err != nil {
				return nil, err
			}
			fg, err := fgStream.next()
			if err != nil {
				return nil, err
			}
			if bitType {
				bg := fg
				if bgPerPattern {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				} else {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				}
				if err := drawBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, &patternBR, fg, bg); err != nil {
					return nil, err
				}
			} else {
				if !bgPerPattern {
					// legacy mode stores BG per block; consume to stay in sync
					if _, err := bgStream.next(); err != nil {
						return nil, err
					}
				}
				if err := fillBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, fg); err != nil {
					return nil, err
				}
			}
			blockIndex++
		}
	}

	if blockIndex != int(blockCount) {
		return nil, fmt.Errorf("block count mismatch: used %d of %d", blockIndex, blockCount)
	}
	if fgStream.i != fgStream.n {
		return nil, fmt.Errorf("color stream mismatch: fg used=%d expected=%d", fgStream.i, fgStream.n)
	}
	if bgStream.i != bgStream.n {
		return nil, fmt.Errorf("color stream mismatch: bg used=%d expected=%d", bgStream.i, bgStream.n)
	}

	return plane, nil
}

type decoderChannelScratch struct {
	plane    []uint8
	macroBig []bool
}

// Decoder reuses large scratch buffers across Decode calls to reduce allocations.
// It is not safe for concurrent use. The returned *image.RGBA is reused and will
// be overwritten on the next Decode call.
type Decoder struct {
	// Parallel enables internal goroutines (per-channel decode and RGB conversion).
	// Set to false to reduce goroutine overhead and allocations.
	Parallel bool

	payload []byte
	zdec    *zstd.Decoder

	y  decoderChannelScratch
	cb decoderChannelScratch
	cr decoderChannelScratch

	neutral []uint8
	dst     *image.RGBA
}

func NewDecoder() *Decoder {
	return &Decoder{Parallel: true, zdec: mustNewZstdDecoder()}
}

func decodeChannelToPix(data []byte, imgW, imgH int, pix []byte, strideBytes int, channelOffset int) error {
	pos := 0

	readU32 := func(label string) (uint32, error) {
		if len(data)-pos < 4 {
			return 0, fmt.Errorf("decodeChannel: truncated while reading %s", label)
		}
		v := binary.BigEndian.Uint32(data[pos : pos+4])
		pos += 4
		return v, nil
	}

	readSlice := func(n uint32, label string) ([]byte, error) {
		if n > uint32(len(data)-pos) {
			return nil, fmt.Errorf("decodeChannel: truncated while reading %s", label)
		}
		end := pos + int(n)
		s := data[pos:end]
		pos = end
		return s, nil
	}

	blockCount, err := readU32("blockCount")
	if err != nil {
		return err
	}

	sizeStreamLen, err := readU32("sizeStreamLen")
	if err != nil {
		return err
	}
	sizeBytes, err := readSlice(sizeStreamLen, "sizeStream")
	if err != nil {
		return err
	}
	sizeBR := newBitReader(sizeBytes)

	typeStreamLen, err := readU32("typeStreamLen")
	if err != nil {
		return err
	}
	typeBytes, err := readSlice(typeStreamLen, "typeStream")
	if err != nil {
		return err
	}
	typeBR := newBitReader(typeBytes)

	patternLen, err := readU32("patternLen")
	if err != nil {
		return err
	}
	patternBytes, err := readSlice(patternLen, "patternStream")
	if err != nil {
		return err
	}
	patternBR := newBitReader(patternBytes)

	// FG: delta-coded bytes, one value per block.
	fgPackedLen, err := readU32("FG packedLen")
	if err != nil {
		return err
	}
	fgPacked, err := readSlice(fgPackedLen, "FG packed data")
	if err != nil {
		return err
	}
	if blockCount > 0 && fgPackedLen < blockCount {
		return fmt.Errorf("decodeChannel: FG packed data too short")
	}
	fgStream, err := newDeltaStream(fgPacked, int(blockCount))
	if err != nil {
		return err
	}

	// BG: delta-coded bytes, one value per pattern block (type bit = 1).
	bgPerPattern := true
	bgPackedLen, err := readU32("BG packedLen")
	if err != nil {
		return err
	}
	bgPacked, err := readSlice(bgPackedLen, "BG packed data")
	if err != nil {
		return err
	}
	if bgPackedLen > blockCount {
		return fmt.Errorf("decodeChannel: BG packed data too long")
	}
	bgStream, err := newDeltaStream(bgPacked, int(bgPackedLen))
	if err != nil {
		return err
	}

	w4 := (imgW / smallBlock) * smallBlock
	h4 := (imgH / smallBlock) * smallBlock
	fullW := (w4 / macroBlock) * macroBlock
	fullH := (h4 / macroBlock) * macroBlock
	useMacro := macroBlock > smallBlock

	blockIndex := 0

	// Hot path for the most common benchmark setting (quality >= 80):
	// - macro blocks are 2x2
	// - small blocks are 1x1 (always solid)
	// BG is stored only for pattern blocks (type bit = 1), otherwise BG == FG.
	if useMacro && smallBlock == 1 && macroBlock == 2 {
		macroGroupsX := fullW / 2
		macroGroupsY := fullH / 2
		macroGroupCount := macroGroupsX * macroGroupsY

		if macroGroupCount > 0 && len(sizeBytes)*8 < macroGroupCount {
			return fmt.Errorf("decodeChannel: size stream too short")
		}
		if int(blockCount) > 0 && len(typeBytes)*8 < int(blockCount) {
			return fmt.Errorf("decodeChannel: type stream too short")
		}

		for my := 0; my < fullH; my += 2 {
			row0 := my*strideBytes + channelOffset
			row1 := row0 + strideBytes
			for mx := 0; mx < fullW; mx += 2 {
				if blockIndex >= int(blockCount) {
					return fmt.Errorf("unexpected end of blocks in main area")
				}

				macroIsBig := sizeBR.readBitFast()
				if macroIsBig {
					bitType := typeBR.readBitFast()
					fg := fgStream.nextFast()

					o00 := row0 + mx*4
					o01 := o00 + 4
					o10 := row1 + mx*4
					o11 := o10 + 4
					if o00 < 0 || o11 >= len(pix) {
						return fmt.Errorf("decodeChannel: index out of range")
					}

					if bitType {
						if bgStream.i >= bgStream.n {
							return fmt.Errorf("decodeChannel: BG stream truncated")
						}
						bg := bgStream.nextFast()

						bits, err := patternBR.readBits(4)
						if err != nil {
							return err
						}

						if (bits & 0b1000) != 0 {
							pix[o00] = fg
						} else {
							pix[o00] = bg
						}
						if (bits & 0b0100) != 0 {
							pix[o01] = fg
						} else {
							pix[o01] = bg
						}
						if (bits & 0b0010) != 0 {
							pix[o10] = fg
						} else {
							pix[o10] = bg
						}
						if (bits & 0b0001) != 0 {
							pix[o11] = fg
						} else {
							pix[o11] = bg
						}
					} else {
						pix[o00] = fg
						pix[o01] = fg
						pix[o10] = fg
						pix[o11] = fg
					}

					blockIndex++
					continue
				}

				_ = typeBR.readBitsFast(4) // four 1x1 blocks; type bits are all 0

				v0 := fgStream.nextFast()
				v1 := fgStream.nextFast()
				v2 := fgStream.nextFast()
				v3 := fgStream.nextFast()

				o00 := row0 + mx*4
				o01 := o00 + 4
				o10 := row1 + mx*4
				o11 := o10 + 4
				if o00 < 0 || o11 >= len(pix) {
					return fmt.Errorf("decodeChannel: index out of range")
				}
				pix[o00] = v0
				pix[o01] = v1
				pix[o10] = v2
				pix[o11] = v3

				blockIndex += 4
			}
		}

		// right stripe: 1x1 blocks only
		for my := 0; my < fullH; my++ {
			row := my*strideBytes + channelOffset
			for mx := fullW; mx < w4; mx++ {
				_ = typeBR.readBitFast()
				v := fgStream.nextFast()

				o := row + mx*4
				if o < 0 || o >= len(pix) {
					return fmt.Errorf("decodeChannel: index out of range")
				}
				pix[o] = v
				blockIndex++
			}
		}

		// bottom stripe: 1x1 blocks only (including bottom-right corner)
		for my := fullH; my < h4; my++ {
			row := my*strideBytes + channelOffset
			for mx := 0; mx < w4; mx++ {
				_ = typeBR.readBitFast()
				v := fgStream.nextFast()

				o := row + mx*4
				if o < 0 || o >= len(pix) {
					return fmt.Errorf("decodeChannel: index out of range")
				}
				pix[o] = v
				blockIndex++
			}
		}

		if blockIndex != int(blockCount) {
			return fmt.Errorf("block count mismatch: used %d of %d", blockIndex, blockCount)
		}
		if fgStream.i != fgStream.n {
			return fmt.Errorf("color stream mismatch: fg used=%d expected=%d", fgStream.i, fgStream.n)
		}
		if bgStream.i != bgStream.n {
			return fmt.Errorf("color stream mismatch: bg used=%d expected=%d", bgStream.i, bgStream.n)
		}
		return nil
	}

	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			if blockIndex >= int(blockCount) {
				return fmt.Errorf("unexpected end of blocks in main area")
			}

			macroIsBig, err := sizeBR.readBit()
			if err != nil {
				return err
			}

			if useMacro && macroIsBig {
				bitType, err := typeBR.readBit()
				if err != nil {
					return err
				}
				fg := fgStream.nextFast()
				if bitType {
					var bg uint8
					if bgPerPattern {
						if bgStream.i >= bgStream.n {
							return fmt.Errorf("decodeChannel: BG stream truncated")
						}
						bg = bgStream.nextFast()
					} else {
						bg = bgStream.nextFast()
					}
					if err := drawBlockPix(pix, strideBytes, mx, my, macroBlock, macroBlock, &patternBR, fg, bg, channelOffset); err != nil {
						return err
					}
				} else {
					if !bgPerPattern {
						_ = bgStream.nextFast()
					}
					if err := fillBlockPix(pix, strideBytes, mx, my, macroBlock, macroBlock, fg, channelOffset); err != nil {
						return err
					}
				}
				blockIndex++
				continue
			}

			for by := 0; by < macroBlock; by += smallBlock {
				for bx := 0; bx < macroBlock; bx += smallBlock {
					if blockIndex >= int(blockCount) {
						return fmt.Errorf("unexpected end of blocks in macro grid")
					}
					bitType, err := typeBR.readBit()
					if err != nil {
						return err
					}
					fg := fgStream.nextFast()
					if bitType {
						var bg uint8
						if bgPerPattern {
							if bgStream.i >= bgStream.n {
								return fmt.Errorf("decodeChannel: BG stream truncated")
							}
							bg = bgStream.nextFast()
						} else {
							bg = bgStream.nextFast()
						}
						if err := drawBlockPix(pix, strideBytes, mx+bx, my+by, smallBlock, smallBlock, &patternBR, fg, bg, channelOffset); err != nil {
							return err
						}
					} else {
						if !bgPerPattern {
							_ = bgStream.nextFast()
						}
						if err := fillBlockPix(pix, strideBytes, mx+bx, my+by, smallBlock, smallBlock, fg, channelOffset); err != nil {
							return err
						}
					}
					blockIndex++
				}
			}
		}
	}

	// right stripe: small blocks only
	for my := 0; my < fullH; my += smallBlock {
		for mx := fullW; mx < w4; mx += smallBlock {
			if blockIndex >= int(blockCount) {
				return fmt.Errorf("unexpected end of blocks in right stripe")
			}
			bitType, err := typeBR.readBit()
			if err != nil {
				return err
			}
			fg := fgStream.nextFast()
			if bitType {
				var bg uint8
				if bgPerPattern {
					if bgStream.i >= bgStream.n {
						return fmt.Errorf("decodeChannel: BG stream truncated")
					}
					bg = bgStream.nextFast()
				} else {
					bg = bgStream.nextFast()
				}
				if err := drawBlockPix(pix, strideBytes, mx, my, smallBlock, smallBlock, &patternBR, fg, bg, channelOffset); err != nil {
					return err
				}
			} else {
				if !bgPerPattern {
					_ = bgStream.nextFast()
				}
				if err := fillBlockPix(pix, strideBytes, mx, my, smallBlock, smallBlock, fg, channelOffset); err != nil {
					return err
				}
			}
			blockIndex++
		}
	}

	// bottom stripe: small blocks only (including bottom-right corner)
	for my := fullH; my < h4; my += smallBlock {
		for mx := 0; mx < w4; mx += smallBlock {
			if blockIndex >= int(blockCount) {
				return fmt.Errorf("unexpected end of blocks in bottom stripe")
			}
			bitType, err := typeBR.readBit()
			if err != nil {
				return err
			}
			fg := fgStream.nextFast()
			if bitType {
				var bg uint8
				if bgPerPattern {
					if bgStream.i >= bgStream.n {
						return fmt.Errorf("decodeChannel: BG stream truncated")
					}
					bg = bgStream.nextFast()
				} else {
					bg = bgStream.nextFast()
				}
				if err := drawBlockPix(pix, strideBytes, mx, my, smallBlock, smallBlock, &patternBR, fg, bg, channelOffset); err != nil {
					return err
				}
			} else {
				if !bgPerPattern {
					_ = bgStream.nextFast()
				}
				if err := fillBlockPix(pix, strideBytes, mx, my, smallBlock, smallBlock, fg, channelOffset); err != nil {
					return err
				}
			}
			blockIndex++
		}
	}

	if blockIndex != int(blockCount) {
		return fmt.Errorf("block count mismatch: used %d of %d", blockIndex, blockCount)
	}
	if fgStream.i != fgStream.n {
		return fmt.Errorf("color stream mismatch: fg used=%d expected=%d", fgStream.i, fgStream.n)
	}
	if bgStream.i != bgStream.n {
		return fmt.Errorf("color stream mismatch: bg used=%d expected=%d", bgStream.i, bgStream.n)
	}
	return nil
}

func (d *Decoder) decodeChannelInto(data []byte, imgW, imgH int, scratch *decoderChannelScratch) ([]uint8, error) {
	pos := 0

	// helpers to read from the raw byte slice without extra allocations
	readU32 := func(label string) (uint32, error) {
		if len(data)-pos < 4 {
			return 0, fmt.Errorf("decodeChannel: truncated while reading %s", label)
		}
		v := binary.BigEndian.Uint32(data[pos : pos+4])
		pos += 4
		return v, nil
	}

	readSlice := func(n uint32, label string) ([]byte, error) {
		if n > uint32(len(data)-pos) {
			return nil, fmt.Errorf("decodeChannel: truncated while reading %s", label)
		}
		end := pos + int(n)
		s := data[pos:end]
		pos = end
		return s, nil
	}

	// blockCount
	blockCount, err := readU32("blockCount")
	if err != nil {
		return nil, err
	}

	// size stream (bit flags for macro vs small in the main area)
	sizeStreamLen, err := readU32("sizeStreamLen")
	if err != nil {
		return nil, err
	}
	sizeBytes, err := readSlice(sizeStreamLen, "sizeStream")
	if err != nil {
		return nil, err
	}
	sizeBR := newBitReader(sizeBytes)

	// type stream (pattern vs solid blocks)
	typeStreamLen, err := readU32("typeStreamLen")
	if err != nil {
		return nil, err
	}
	typeBytes, err := readSlice(typeStreamLen, "typeStream")
	if err != nil {
		return nil, err
	}
	typeBR := newBitReader(typeBytes)

	// pattern stream (actual bi-level patterns)
	patternLen, err := readU32("patternLen")
	if err != nil {
		return nil, err
	}
	patternBytes, err := readSlice(patternLen, "patternStream")
	if err != nil {
		return nil, err
	}
	patternBR := newBitReader(patternBytes)

	// FG: delta-coded bytes, one value per block.
	fgPackedLen, err := readU32("FG packedLen")
	if err != nil {
		return nil, err
	}
	fgPacked, err := readSlice(fgPackedLen, "FG packed data")
	if err != nil {
		return nil, err
	}
	if blockCount > 0 && fgPackedLen < blockCount {
		return nil, fmt.Errorf("decodeChannel: FG packed data too short")
	}
	fgStream, err := newDeltaStream(fgPacked, int(blockCount))
	if err != nil {
		return nil, err
	}

	// BG: delta-coded bytes, one value per pattern block (type bit = 1).
	bgPerPattern := true
	bgPackedLen, err := readU32("BG packedLen")
	if err != nil {
		return nil, err
	}
	bgPacked, err := readSlice(bgPackedLen, "BG packed data")
	if err != nil {
		return nil, err
	}
	if bgPackedLen > blockCount {
		return nil, fmt.Errorf("decodeChannel: BG packed data too long")
	}
	bgStream, err := newDeltaStream(bgPacked, int(bgPackedLen))
	if err != nil {
		return nil, err
	}

	// reconstruct the block geometry and fill the plane
	needPlane := imgW * imgH
	if cap(scratch.plane) < needPlane {
		scratch.plane = make([]uint8, needPlane)
	} else {
		scratch.plane = scratch.plane[:needPlane]
	}
	plane := scratch.plane

	w4 := (imgW / smallBlock) * smallBlock
	h4 := (imgH / smallBlock) * smallBlock
	fullW := (w4 / macroBlock) * macroBlock
	fullH := (h4 / macroBlock) * macroBlock
	macroGroupCount := (fullW / macroBlock) * (fullH / macroBlock)
	useMacro := macroBlock > smallBlock

	if macroGroupCount > 0 {
		if cap(scratch.macroBig) < macroGroupCount {
			scratch.macroBig = make([]bool, macroGroupCount)
		} else {
			scratch.macroBig = scratch.macroBig[:macroGroupCount]
		}
	} else {
		scratch.macroBig = scratch.macroBig[:0]
	}
	macroBig := scratch.macroBig

	// rebuild macro-block size decisions from the size stream
	for i := range macroGroupCount {
		bit, err := sizeBR.readBit()
		if err != nil {
			return nil, err
		}
		macroBig[i] = bit
	}

	blockIndex := 0
	macroIndex := 0

	// main macroBlock x macroBlock area
	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			if blockIndex >= int(blockCount) {
				return nil, fmt.Errorf("unexpected end of blocks in main area")
			}
			if useMacro && macroBig[macroIndex] {
				// macroBlock: read one type bit, FG and BG from the delta streams, use full pattern
				bitType, err := typeBR.readBit()
				if err != nil {
					return nil, err
				}

				fg, err := fgStream.next()
				if err != nil {
					return nil, err
				}
				if bitType {
					bg := fg
					if bgPerPattern {
						bg, err = bgStream.next()
						if err != nil {
							return nil, err
						}
					} else {
						bg, err = bgStream.next()
						if err != nil {
							return nil, err
						}
					}
					if err := drawBlockPlane(plane, imgW, mx, my, macroBlock, macroBlock, &patternBR, fg, bg); err != nil {
						return nil, err
					}
				} else {
					if !bgPerPattern {
						// legacy mode stores BG per block; consume to stay in sync
						if _, err := bgStream.next(); err != nil {
							return nil, err
						}
					}
					if err := fillBlockPlane(plane, imgW, mx, my, macroBlock, macroBlock, fg); err != nil {
						return nil, err
					}
				}
				blockIndex++
			} else {
				// grid of smallBlock x smallBlock blocks covering the macroBlock area
				for by := 0; by < macroBlock; by += smallBlock {
					for bx := 0; bx < macroBlock; bx += smallBlock {
						if blockIndex >= int(blockCount) {
							return nil, fmt.Errorf("unexpected end of blocks in macro grid")
						}
						bitType, err := typeBR.readBit()
						if err != nil {
							return nil, err
						}
						fg, err := fgStream.next()
						if err != nil {
							return nil, err
						}
						if bitType {
							bg := fg
							if bgPerPattern {
								bg, err = bgStream.next()
								if err != nil {
									return nil, err
								}
							} else {
								bg, err = bgStream.next()
								if err != nil {
									return nil, err
								}
							}
							if err := drawBlockPlane(plane, imgW, mx+bx, my+by, smallBlock, smallBlock, &patternBR, fg, bg); err != nil {
								return nil, err
							}
						} else {
							if !bgPerPattern {
								// legacy mode stores BG per block; consume to stay in sync
								if _, err := bgStream.next(); err != nil {
									return nil, err
								}
							}
							if err := fillBlockPlane(plane, imgW, mx+bx, my+by, smallBlock, smallBlock, fg); err != nil {
								return nil, err
							}
						}
						blockIndex++
					}
				}
			}
			macroIndex++
		}
	}

	// right stripe: small blocks only
	for my := 0; my < fullH; my += smallBlock {
		for mx := fullW; mx < w4; mx += smallBlock {
			if blockIndex >= int(blockCount) {
				return nil, fmt.Errorf("unexpected end of blocks in right stripe")
			}
			bitType, err := typeBR.readBit()
			if err != nil {
				return nil, err
			}
			fg, err := fgStream.next()
			if err != nil {
				return nil, err
			}
			if bitType {
				bg := fg
				if bgPerPattern {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				} else {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				}
				if err := drawBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, &patternBR, fg, bg); err != nil {
					return nil, err
				}
			} else {
				if !bgPerPattern {
					// legacy mode stores BG per block; consume to stay in sync
					if _, err := bgStream.next(); err != nil {
						return nil, err
					}
				}
				if err := fillBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, fg); err != nil {
					return nil, err
				}
			}
			blockIndex++
		}
	}
	// bottom stripe: small blocks only (including bottom-right corner)
	for my := fullH; my < h4; my += smallBlock {
		for mx := 0; mx < w4; mx += smallBlock {
			if blockIndex >= int(blockCount) {
				return nil, fmt.Errorf("unexpected end of blocks in bottom stripe")
			}
			bitType, err := typeBR.readBit()
			if err != nil {
				return nil, err
			}
			fg, err := fgStream.next()
			if err != nil {
				return nil, err
			}
			if bitType {
				bg := fg
				if bgPerPattern {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				} else {
					bg, err = bgStream.next()
					if err != nil {
						return nil, err
					}
				}
				if err := drawBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, &patternBR, fg, bg); err != nil {
					return nil, err
				}
			} else {
				if !bgPerPattern {
					// legacy mode stores BG per block; consume to stay in sync
					if _, err := bgStream.next(); err != nil {
						return nil, err
					}
				}
				if err := fillBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, fg); err != nil {
					return nil, err
				}
			}
			blockIndex++
		}
	}

	if blockIndex != int(blockCount) {
		return nil, fmt.Errorf("block count mismatch: used %d of %d", blockIndex, blockCount)
	}
	if fgStream.i != fgStream.n {
		return nil, fmt.Errorf("color stream mismatch: fg used=%d expected=%d", fgStream.i, fgStream.n)
	}
	if bgStream.i != bgStream.n {
		return nil, fmt.Errorf("color stream mismatch: bg used=%d expected=%d", bgStream.i, bgStream.n)
	}

	return plane, nil
}

func (d *Decoder) Decode(compData []byte, postfilter bool) (*image.RGBA, error) {
	if d.zdec == nil {
		d.zdec = mustNewZstdDecoder()
	}
	payload, err := d.zdec.DecodeAll(compData, d.payload[:0])
	if err != nil {
		return nil, fmt.Errorf("zstd decode: %w", err)
	}
	d.payload = payload

	if len(payload) < len(codec) {
		return nil, fmt.Errorf("read header: short magic")
	}
	pos := 0

	// magic
	if string(payload[pos:pos+len(codec)]) != codec {
		return nil, fmt.Errorf("bad magic: %q", string(payload[pos:pos+len(codec)]))
	}
	pos += len(codec)

	readU16 := func(label string) (uint16, error) {
		if len(payload)-pos < 2 {
			return 0, fmt.Errorf("decode: truncated while reading %s", label)
		}
		v := binary.BigEndian.Uint16(payload[pos : pos+2])
		pos += 2
		return v, nil
	}
	readU32 := func(label string) (uint32, error) {
		if len(payload)-pos < 4 {
			return 0, fmt.Errorf("decode: truncated while reading %s", label)
		}
		v := binary.BigEndian.Uint32(payload[pos : pos+4])
		pos += 4
		return v, nil
	}

	bwSize, err := readU16("block width")
	if err != nil {
		return nil, err
	}
	bhSize, err := readU16("block height")
	if err != nil {
		return nil, err
	}
	if bwSize == 0 || bhSize == 0 {
		return nil, fmt.Errorf("invalid block sizes in header: %dx%d", bwSize, bhSize)
	}
	smallBlock = int(bwSize)
	macroBlock = int(bhSize)

	// channels mask: which Y/Cb/Cr planes are stored.
	if len(payload)-pos < 1 {
		return nil, fmt.Errorf("decode: truncated while reading channels mask")
	}
	channelsMask := payload[pos]
	pos++
	if channelsMask&channelFlagY == 0 {
		return nil, fmt.Errorf("decode: Y channel missing in header")
	}

	imgW32, err := readU32("image width")
	if err != nil {
		return nil, err
	}
	imgH32, err := readU32("image height")
	if err != nil {
		return nil, err
	}
	imgW := int(imgW32)
	imgH := int(imgH32)

	if macroBlock < smallBlock || macroBlock%smallBlock != 0 {
		return nil, fmt.Errorf("macroBlock (%d) must be >= smallBlock (%d) and a multiple of it",
			macroBlock, smallBlock)
	}

	if d.dst == nil || d.dst.Bounds().Dx() != imgW || d.dst.Bounds().Dy() != imgH {
		d.dst = image.NewRGBA(image.Rect(0, 0, imgW, imgH))
	}
	dst := d.dst
	pix := dst.Pix
	stride := dst.Stride

	ySeg, err := readChannelSegment(payload, &pos)
	if err != nil {
		return nil, err
	}
	hasCb := (channelsMask & channelFlagCb) != 0
	hasCr := (channelsMask & channelFlagCr) != 0

	var cbSeg, crSeg []byte
	if hasCb {
		cbSeg, err = readChannelSegment(payload, &pos)
		if err != nil {
			return nil, err
		}
	}
	if hasCr {
		crSeg, err = readChannelSegment(payload, &pos)
		if err != nil {
			return nil, err
		}
	}

	w4 := (imgW / smallBlock) * smallBlock
	h4 := (imgH / smallBlock) * smallBlock
	if w4 != imgW || h4 != imgH {
		for o := 0; o+3 < len(pix); o += 4 {
			pix[o+0] = 0
			pix[o+1] = 128
			pix[o+2] = 128
			pix[o+3] = 255
		}
	}

	var errY, errCb, errCr error
	if d.Parallel {
		var wg sync.WaitGroup
		wg.Add(1)
		go decodeChannelToPixWorker(ySeg, imgW, imgH, pix, stride, 0, &errY, &wg)
		if hasCb {
			wg.Add(1)
			go decodeChannelToPixWorker(cbSeg, imgW, imgH, pix, stride, 1, &errCb, &wg)
		}
		if hasCr {
			wg.Add(1)
			go decodeChannelToPixWorker(crSeg, imgW, imgH, pix, stride, 2, &errCr, &wg)
		}
		wg.Wait()
	} else {
		errY = decodeChannelToPix(ySeg, imgW, imgH, pix, stride, 0)
		if hasCb {
			errCb = decodeChannelToPix(cbSeg, imgW, imgH, pix, stride, 1)
		}
		if hasCr {
			errCr = decodeChannelToPix(crSeg, imgW, imgH, pix, stride, 2)
		}
	}

	if errY != nil {
		return nil, errY
	}
	if errCb != nil {
		return nil, errCb
	}
	if errCr != nil {
		return nil, errCr
	}

	if d.Parallel {
		workers := max(min(runtime.NumCPU(), imgH), 1)
		rowsPerWorker := (imgH + workers - 1) / workers

		var wgRGB sync.WaitGroup
		for i := range workers {
			y0 := i * rowsPerWorker
			if y0 >= imgH {
				break
			}
			y1 := min(y0+rowsPerWorker, imgH)

			wgRGB.Add(1)
			go ycbcrToRGBStripe(pix, stride, imgW, y0, y1, hasCb, hasCr, &wgRGB)
		}
		wgRGB.Wait()
	} else {
		ycbcrToRGB(pix, stride, imgW, 0, imgH, hasCb, hasCr)
	}

	if postfilter {
		return smoothBlocks(dst), nil
	}

	return dst, nil
}

// DecodeFrom reads compressed data from r and decodes it.
// This mirrors codecs that accept io.Reader/io.Writer and is useful for
// benchmarking. It allocates to read the full input; for zero-copy decoding,
// prefer Decode([]byte,...).
func (d *Decoder) DecodeFrom(r io.Reader, postfilter bool) (*image.RGBA, error) {
	compData, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return d.Decode(compData, postfilter)
}

func decodeChannelToPixWorker(data []byte, imgW, imgH int, pix []byte, strideBytes int, channelOffset int, dstErr *error, wg *sync.WaitGroup) {
	defer wg.Done()
	*dstErr = decodeChannelToPix(data, imgW, imgH, pix, strideBytes, channelOffset)
}

func ycbcrToRGB(pix []byte, stride, imgW int, yStart, yEnd int, hasCb, hasCr bool) {
	if hasCb && hasCr {
		for y := yStart; y < yEnd; y++ {
			rowOff := y * stride
			for x := 0; x < imgW; x++ {
				o := rowOff + x*4
				Y := int32(pix[o+0])
				Cb := int32(pix[o+1]) - 128
				Cr := int32(pix[o+2]) - 128

				R := Y + ((91881 * Cr) >> 16)
				G := Y - ((22554*Cb + 46802*Cr) >> 16)
				B := Y + ((116130 * Cb) >> 16)

				if R < 0 {
					R = 0
				} else if R > 255 {
					R = 255
				}
				if G < 0 {
					G = 0
				} else if G > 255 {
					G = 255
				}
				if B < 0 {
					B = 0
				} else if B > 255 {
					B = 255
				}

				pix[o+0] = uint8(R)
				pix[o+1] = uint8(G)
				pix[o+2] = uint8(B)
				pix[o+3] = 255
			}
		}
		return
	}

	if hasCb {
		for y := yStart; y < yEnd; y++ {
			rowOff := y * stride
			for x := 0; x < imgW; x++ {
				o := rowOff + x*4
				Y := int32(pix[o+0])
				Cb := int32(pix[o+1]) - 128

				R := Y
				G := Y - ((22554 * Cb) >> 16)
				B := Y + ((116130 * Cb) >> 16)

				if R < 0 {
					R = 0
				} else if R > 255 {
					R = 255
				}
				if G < 0 {
					G = 0
				} else if G > 255 {
					G = 255
				}
				if B < 0 {
					B = 0
				} else if B > 255 {
					B = 255
				}

				pix[o+0] = uint8(R)
				pix[o+1] = uint8(G)
				pix[o+2] = uint8(B)
				pix[o+3] = 255
			}
		}
		return
	}

	if hasCr {
		for y := yStart; y < yEnd; y++ {
			rowOff := y * stride
			for x := 0; x < imgW; x++ {
				o := rowOff + x*4
				Y := int32(pix[o+0])
				Cr := int32(pix[o+2]) - 128

				R := Y + ((91881 * Cr) >> 16)
				G := Y - ((46802 * Cr) >> 16)
				B := Y

				if R < 0 {
					R = 0
				} else if R > 255 {
					R = 255
				}
				if G < 0 {
					G = 0
				} else if G > 255 {
					G = 255
				}
				if B < 0 {
					B = 0
				} else if B > 255 {
					B = 255
				}

				pix[o+0] = uint8(R)
				pix[o+1] = uint8(G)
				pix[o+2] = uint8(B)
				pix[o+3] = 255
			}
		}
		return
	}

	// Y only.
	for y := yStart; y < yEnd; y++ {
		rowOff := y * stride
		for x := 0; x < imgW; x++ {
			o := rowOff + x*4
			Y := pix[o+0]
			pix[o+0] = Y
			pix[o+1] = Y
			pix[o+2] = Y
			pix[o+3] = 255
		}
	}
}

func ycbcrToRGBStripe(pix []byte, stride, imgW int, yStart, yEnd int, hasCb, hasCr bool, wg *sync.WaitGroup) {
	defer wg.Done()
	ycbcrToRGB(pix, stride, imgW, yStart, yEnd, hasCb, hasCr)
}

// Decode reads a BABE-compressed stream (Zstd + three independent Y/Cb/Cr streams)
// and returns an image.Image.
func Decode(compData []byte, postfilter bool) (image.Image, error) {
	d := NewDecoder()
	return d.Decode(compData, postfilter)
}

// smoothJunctions performs gradient-based smoothing at intersections of small blocks.
// At each internal junction of the smallBlock grid, it inspects the four corner pixels
// and, if they are luminance-wise similar, fills a small rectangle around the junction
// with a bilinear gradient between the four corner colors.
func smoothJunctions(img *image.RGBA) *image.RGBA {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	// need at least 2x2 blocks to have internal junctions
	if w < 2*smallBlock || h < 2*smallBlock {
		return img
	}

	maxRadius := min(smallBlock, 3)
	const junctionLumaSpread int32 = 32
	const lumaSimThreshold int32 = 8

	minX := b.Min.X
	minY := b.Min.Y
	stride := img.Stride
	srcPix := img.Pix

	// Work on a copy so that all reads come from the original image
	// and only writes go to the destination buffer.
	dst := image.NewRGBA(b)
	copy(dst.Pix, img.Pix)
	dstPix := dst.Pix

	// helper to compute luma from 8-bit RGB directly
	luma8 := func(r, g, b uint8) int32 {
		rr := int32(r)
		gg := int32(g)
		bb := int32(b)
		// Integer approximation of ITU-R BT.601:
		// Y ≈ 0.299 R + 0.587 G + 0.114 B
		return (299*rr + 587*gg + 114*bb) / 1000
	}

	similar := func(r1, g1, b1, a1, r2, g2, b2, a2 uint8) bool {
		da := luma8(r1, g1, b1) - luma8(r2, g2, b2)
		if da < 0 {
			da = -da
		}
		return da <= lumaSimThreshold
	}

	lerp8 := func(a, b uint8, t float64) uint8 {
		return uint8(float64(a)*(1-t) + float64(b)*t + 0.5)
	}

	// Parallelise over horizontal stripes. Each worker owns a disjoint band
	// of rows in the destination image. Writes are clamped to the worker's
	// stripe to avoid concurrent writes to the same pixels.
	workers := max(min(runtime.NumCPU(), h), 1)
	rowsPerWorker := (h + workers - 1) / workers

	var wg sync.WaitGroup
	for i := range workers {
		stripeStart := b.Min.Y + i*rowsPerWorker
		if stripeStart >= b.Max.Y {
			break
		}
		stripeEnd := min(stripeStart+rowsPerWorker, b.Max.Y)

		wg.Add(1)
		go func(yStripeStart, yStripeEnd int) {
			defer wg.Done()

			for y := b.Min.Y + smallBlock; y < b.Max.Y; y += smallBlock {
				// Only process junctions whose center row lies inside this stripe.
				if y < yStripeStart || y >= yStripeEnd {
					continue
				}

				for x := b.Min.X + smallBlock; x < b.Max.X; x += smallBlock {
					// four corners around the junction, read from the original image
					idx00 := (y-1-minY)*stride + (x-1-minX)*4 // top-left
					idx10 := (y-1-minY)*stride + (x-minX)*4   // top-right
					idx01 := (y-minY)*stride + (x-1-minX)*4   // bottom-left
					idx11 := (y-minY)*stride + (x-minX)*4     // bottom-right

					r00, g00, b00, a00 := srcPix[idx00+0], srcPix[idx00+1], srcPix[idx00+2], srcPix[idx00+3]
					r10, g10, b10, a10 := srcPix[idx10+0], srcPix[idx10+1], srcPix[idx10+2], srcPix[idx10+3]
					r01, g01, b01, a01 := srcPix[idx01+0], srcPix[idx01+1], srcPix[idx01+2], srcPix[idx01+3]
					r11, g11, b11, a11 := srcPix[idx11+0], srcPix[idx11+1], srcPix[idx11+2], srcPix[idx11+3]

					l00 := luma8(r00, g00, b00)
					l10 := luma8(r10, g10, b10)
					l01 := luma8(r01, g01, b01)
					l11 := luma8(r11, g11, b11)

					minL, maxL := l00, l00
					for _, v := range []int32{l10, l01, l11} {
						if v < minL {
							minL = v
						}
						if v > maxL {
							maxL = v
						}
					}
					// if the corner luminance differs too much, treat it as a hard edge and skip
					if maxL-minL > junctionLumaSpread {
						continue
					}

					// grow small rectangle around the junction while colors remain similar
					L := 0
					for i := 1; i <= maxRadius && x-i >= b.Min.X; i++ {
						idxUp := (y-1-minY)*stride + (x-i-minX)*4
						idxDown := (y-minY)*stride + (x-i-minX)*4
						rUp, gUp, bUp, aUp := srcPix[idxUp+0], srcPix[idxUp+1], srcPix[idxUp+2], srcPix[idxUp+3]
						rDown, gDown, bDown, aDown := srcPix[idxDown+0], srcPix[idxDown+1], srcPix[idxDown+2], srcPix[idxDown+3]
						if !similar(rUp, gUp, bUp, aUp, r00, g00, b00, a00) || !similar(rDown, gDown, bDown, aDown, r01, g01, b01, a01) {
							break
						}
						L = i
					}

					R := 0
					for i := 1; i <= maxRadius && x-1+i < b.Max.X; i++ {
						idxUp := (y-1-minY)*stride + (x-1+i-minX)*4
						idxDown := (y-minY)*stride + (x-1+i-minX)*4
						rUp, gUp, bUp, aUp := srcPix[idxUp+0], srcPix[idxUp+1], srcPix[idxUp+2], srcPix[idxUp+3]
						rDown, gDown, bDown, aDown := srcPix[idxDown+0], srcPix[idxDown+1], srcPix[idxDown+2], srcPix[idxDown+3]
						if !similar(rUp, gUp, bUp, aUp, r10, g10, b10, a10) || !similar(rDown, gDown, bDown, aDown, r11, g11, b11, a11) {
							break
						}
						R = i
					}

					U := 0
					for i := 1; i <= maxRadius && y-i >= b.Min.Y; i++ {
						idxLeft := (y-i-minY)*stride + (x-1-minX)*4
						idxRight := (y-i-minY)*stride + (x-minX)*4
						rLeft, gLeft, bLeft, aLeft := srcPix[idxLeft+0], srcPix[idxLeft+1], srcPix[idxLeft+2], srcPix[idxLeft+3]
						rRight, gRight, bRight, aRight := srcPix[idxRight+0], srcPix[idxRight+1], srcPix[idxRight+2], srcPix[idxRight+3]
						if !similar(rLeft, gLeft, bLeft, aLeft, r00, g00, b00, a00) || !similar(rRight, gRight, bRight, aRight, r10, g10, b10, a10) {
							break
						}
						U = i
					}

					D := 0
					for i := 1; i <= maxRadius && y-1+i < b.Max.Y; i++ {
						idxLeft := (y-1+i-minY)*stride + (x-1-minX)*4
						idxRight := (y-1+i-minY)*stride + (x-minX)*4
						rLeft, gLeft, bLeft, aLeft := srcPix[idxLeft+0], srcPix[idxLeft+1], srcPix[idxLeft+2], srcPix[idxLeft+3]
						rRight, gRight, bRight, aRight := srcPix[idxRight+0], srcPix[idxRight+1], srcPix[idxRight+2], srcPix[idxRight+3]
						if !similar(rLeft, gLeft, bLeft, aLeft, r01, g01, b01, a01) || !similar(rRight, gRight, bRight, aRight, r11, g11, b11, a11) {
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

					if rectMinX < b.Min.X || rectMaxX >= b.Max.X || rectMinY < b.Min.Y || rectMaxY >= b.Max.Y {
						continue
					}
					if rectMaxX <= rectMinX || rectMaxY <= rectMinY {
						continue
					}

					// Clamp vertical extent to the current worker's stripe so that
					// no two goroutines write to the same rows.
					if rectMaxY < yStripeStart || rectMinY >= yStripeEnd {
						continue
					}
					if rectMinY < yStripeStart {
						rectMinY = yStripeStart
					}
					if rectMaxY >= yStripeEnd {
						rectMaxY = yStripeEnd - 1
					}

					width := rectMaxX - rectMinX
					height := rectMaxY - rectMinY

					for py := rectMinY; py <= rectMaxY; py++ {
						v := float64(py-rectMinY) / float64(height)
						rowOff := (py - minY) * stride
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

							idx := rowOff + (px-minX)*4
							dstPix[idx+0] = r
							dstPix[idx+1] = g
							dstPix[idx+2] = bc
							dstPix[idx+3] = a
						}
					}
				}
			}
		}(stripeStart, stripeEnd)
	}
	wg.Wait()

	return dst
}

// smoothFlatAreas performs a light deblocking pass:
// it smooths only along block boundaries where the luminance difference is small.
func smoothFlatAreas(src *image.RGBA) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()

	// if the image is too small for at least two blocks in each direction, just return it as-is
	if w < 2*smallBlock || h < 2*smallBlock {
		return src
	}

	srcPix := src.Pix
	stride := src.Stride

	// threshold on luminance difference across a block boundary
	// smaller values -> less smoothing
	const boundaryLumaThreshold int32 = 10

	// helper to compute luma from 8-bit RGB directly
	luma8 := func(r, g, b uint8) int32 {
		rr := int32(r)
		gg := int32(g)
		bb := int32(b)
		// Integer approximation of ITU-R BT.601:
		// Y ≈ 0.299 R + 0.587 G + 0.114 B
		return (299*rr + 587*gg + 114*bb) / 1000
	}

	minX := b.Min.X
	minY := b.Min.Y

	// vertical block boundaries: x = k * smallBlock
	workersV := min(max(runtime.NumCPU(), 1), h)
	rowsPerWorker := (h + workersV - 1) / workersV

	var wgV sync.WaitGroup
	for i := range workersV {
		y0 := b.Min.Y + i*rowsPerWorker
		if y0 >= b.Max.Y {
			break
		}
		y1 := y0 + rowsPerWorker
		if y1 > b.Max.Y {
			y1 = b.Max.Y
		}

		wgV.Add(1)
		go func(yStart, yEnd int) {
			defer wgV.Done()
			for y := yStart; y < yEnd; y++ {
				rowOff := (y - minY) * stride
				for x := b.Min.X + smallBlock; x < b.Max.X; x += smallBlock {
					xL := x - 1
					idxL := rowOff + (xL-minX)*4
					idxR := rowOff + (x-minX)*4

					rL := srcPix[idxL+0]
					gL := srcPix[idxL+1]
					bL := srcPix[idxL+2]
					aL := srcPix[idxL+3]

					rR := srcPix[idxR+0]
					gR := srcPix[idxR+1]
					bR := srcPix[idxR+2]
					aR := srcPix[idxR+3]

					lL := luma8(rL, gL, bL)
					lR := luma8(rR, gR, bR)
					d := lL - lR
					if d < 0 {
						d = -d
					}

					// if the two sides are similar in luminance, slightly blur across the boundary
					if d <= boundaryLumaThreshold {
						r := uint8((uint16(rL) + uint16(rR)) / 2)
						g := uint8((uint16(gL) + uint16(gR)) / 2)
						bc := uint8((uint16(bL) + uint16(bR)) / 2)
						a := uint8((uint16(aL) + uint16(aR)) / 2)

						srcPix[idxL+0] = r
						srcPix[idxL+1] = g
						srcPix[idxL+2] = bc
						srcPix[idxL+3] = a

						srcPix[idxR+0] = r
						srcPix[idxR+1] = g
						srcPix[idxR+2] = bc
						srcPix[idxR+3] = a
					}
				}
			}
		}(y0, y1)
	}
	wgV.Wait()

	// horizontal block boundaries: y = k * smallBlock
	workersH := min(max(runtime.NumCPU(), 1), w)
	colsPerWorker := (w + workersH - 1) / workersH

	var wgH sync.WaitGroup
	for i := range workersH {
		x0 := b.Min.X + i*colsPerWorker
		if x0 >= b.Max.X {
			break
		}
		x1 := min(x0+colsPerWorker, b.Max.X)

		wgH.Add(1)
		go func(xStart, xEnd int) {
			defer wgH.Done()
			for x := xStart; x < xEnd; x++ {
				colOff := (x - minX) * 4
				for y := b.Min.Y + smallBlock; y < b.Max.Y; y += smallBlock {
					yT := y - 1
					idxT := (yT-minY)*stride + colOff
					idxB := (y-minY)*stride + colOff

					rT := srcPix[idxT+0]
					gT := srcPix[idxT+1]
					bT := srcPix[idxT+2]
					aT := srcPix[idxT+3]

					rB := srcPix[idxB+0]
					gB := srcPix[idxB+1]
					bB := srcPix[idxB+2]
					aB := srcPix[idxB+3]

					lT := luma8(rT, gT, bT)
					lB := luma8(rB, gB, bB)
					d := lT - lB
					if d < 0 {
						d = -d
					}

					if d <= boundaryLumaThreshold {
						r := uint8((uint16(rT) + uint16(rB)) / 2)
						g := uint8((uint16(gT) + uint16(gB)) / 2)
						bc := uint8((uint16(bT) + uint16(bB)) / 2)
						a := uint8((uint16(aT) + uint16(aB)) / 2)

						srcPix[idxT+0] = r
						srcPix[idxT+1] = g
						srcPix[idxT+2] = bc
						srcPix[idxT+3] = a

						srcPix[idxB+0] = r
						srcPix[idxB+1] = g
						srcPix[idxB+2] = bc
						srcPix[idxB+3] = a
					}
				}
			}
		}(x0, x1)
	}
	wgH.Wait()

	return src
}

// smoothBlocks runs both junction smoothing and light deblocking in a fixed order.
// This keeps the post-process logic in one place for Decode().
func smoothBlocks(src *image.RGBA) *image.RGBA {
	// If we are operating at the finest granularity (1x1 blocks),
	// there are no visible block boundaries to smooth.
	if smallBlock <= 1 {
		return src
	}
	// first apply light deblocking along block boundaries,
	// then apply gradient smoothing at block junctions.
	return smoothJunctions(smoothFlatAreas(src))
}

type yuv struct {
	Y  int16
	Cb int16
	Cr int16
}

func rgbToYCbCr(r, g, b uint8) yuv {
	rr, gg, bb := int32(r), int32(g), int32(b)
	Y := (77*rr + 150*gg + 29*bb) >> 8 // ≈ 0.299 0.587 0.114
	Cb := ((-43*rr - 85*gg + 128*bb) >> 8) + 128
	Cr := ((128*rr - 107*gg - 21*bb) >> 8) + 128
	return yuv{int16(Y), int16(Cb), int16(Cr)}
}

// delta‑coding of an 8‑bit value on a 0..255 ring into int8 [-128..127]
func encodeDelta8(prev, curr uint8) int8 {
	diff := int16(curr) - int16(prev)
	if diff < -128 {
		diff += 256
	} else if diff > 127 {
		diff -= 256
	}
	return int8(diff)
}

func decodeDelta8(prev uint8, d int8) uint8 {
	return uint8(int16(prev) + int16(d))
}

// --- ZSTD helpers ---

func mustNewZstdEncoder() *zstd.Encoder {
	enc, err := zstd.NewWriter(
		nil,
		zstd.WithEncoderConcurrency(1),
		zstd.WithEncoderLevel(zstd.SpeedBetterCompression),
		zstd.WithLowerEncoderMem(true),
	)
	if err != nil {
		panic(err)
	}
	return enc
}

func mustNewZstdDecoder() *zstd.Decoder {
	dec, err := zstd.NewReader(
		nil,
		zstd.WithDecoderConcurrency(1),
		zstd.WithDecoderLowmem(true),
	)
	if err != nil {
		panic(err)
	}
	return dec
}

var zstdEncPool = sync.Pool{
	New: func() any {
		return mustNewZstdEncoder()
	},
}

var zstdDecPool = sync.Pool{
	New: func() any {
		return mustNewZstdDecoder()
	},
}

func compressZstd(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	enc := zstdEncPool.Get().(*zstd.Encoder)
	out := enc.EncodeAll(data, nil)
	zstdEncPool.Put(enc)
	return out, nil
}

func compressZstdInto(dst []byte, data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	enc := zstdEncPool.Get().(*zstd.Encoder)
	out := enc.EncodeAll(data, dst[:0])
	zstdEncPool.Put(enc)
	return out, nil
}

func decompressZstd(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	dec := zstdDecPool.Get().(*zstd.Decoder)
	out, err := dec.DecodeAll(data, nil)
	zstdDecPool.Put(dec)
	return out, err
}

func decompressZstdInto(dst []byte, data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	dec := zstdDecPool.Get().(*zstd.Decoder)
	out, err := dec.DecodeAll(data, dst[:0])
	zstdDecPool.Put(dec)
	return out, err
}

// bitsNeeded returns how many bits are required to represent v (0..255).
func bitsNeeded(v int) int {
	if v <= 0 {
		return 1
	}
	bits := 0
	for v > 0 {
		bits++
		v >>= 1
	}
	if bits > 8 {
		return 8
	}
	return bits
}

// writeDeltaPackedBytes encodes src using the same layout as deltaPackBytes,
// but streams directly into w to avoid allocating an intermediate slice.
func writeDeltaPackedBytes(w *bufio.Writer, src []uint8) error {
	n := len(src)
	if n == 0 {
		return nil
	}
	if err := w.WriteByte(src[0]); err != nil {
		return err
	}

	prev := src[0]
	i := 1

	var buf [4096]byte
	for i < n {
		j := 0
		for j < len(buf) && i < n {
			d := encodeDelta8(prev, src[i])
			buf[j] = byte(d)
			prev = src[i]
			i++
			j++
		}
		if _, err := w.Write(buf[:j]); err != nil {
			return err
		}
	}
	return nil
}

func writeU16BE(w *bufio.Writer, v uint16) error {
	var buf [2]byte
	binary.BigEndian.PutUint16(buf[:], v)
	_, err := w.Write(buf[:])
	return err
}

func writeU32BE(w *bufio.Writer, v uint32) error {
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], v)
	_, err := w.Write(buf[:])
	return err
}

// Global scale/bits for chroma-related packing (e.g. Cb/Cr).
// packBits must be enough to represent the range [0 .. 255/packScale].
const packScale = 1
const packBits = 8 // bitsNeeded(255 / packScale) for current scale

func init() {
	// Safety check: ensure packBits is consistent with packScale.
	expected := bitsNeeded(255 / packScale)
	if expected != packBits {
		panic(fmt.Sprintf("packBits (%d) mismatch bitsNeeded(255/packScale)=%d", packBits, expected))
	}
}

// deltaPackBytes encodes a byte slice using circular delta coding on 0..255.
// The output layout is:
//
//	[0]       = first byte (if len(src) > 0)
//	[1..]     = (len(src)-1) delta bytes (encodeDelta8(prev, curr))
//
// The caller is responsible for storing/transmitting the original length.
func deltaPackBytes(src []byte) ([]byte, error) {
	n := len(src)
	if n == 0 {
		return []byte{}, nil
	}

	out := make([]byte, n)
	out[0] = src[0]

	prev := src[0]

	for i := 1; i < n; i++ {
		d := encodeDelta8(prev, src[i])
		out[i] = byte(d)
		prev = src[i]
	}

	return out, nil
}

// deltaStream provides on-the-fly decoding of a delta-packed byte slice
// produced by deltaPackBytes, without allocating a separate output buffer.
type deltaStream struct {
	packed []byte
	n      int // total number of decoded values expected
	i      int // number of values already returned
	prev   byte
}

func newDeltaStream(packed []byte, n int) (deltaStream, error) {
	if n == 0 {
		return deltaStream{}, nil
	}
	if len(packed) < n {
		return deltaStream{}, fmt.Errorf("delta stream truncated: have %d, need %d", len(packed), n)
	}
	return deltaStream{
		packed: packed,
		n:      n,
		i:      0,
		prev:   packed[0],
	}, nil
}

func (ds *deltaStream) next() (byte, error) {
	if ds.n == 0 {
		return 0, io.EOF
	}
	if ds.i >= ds.n {
		return 0, io.EOF
	}
	if ds.i == 0 {
		ds.i = 1
		return ds.prev, nil
	}
	d := int8(ds.packed[ds.i])
	v := decodeDelta8(ds.prev, d)
	ds.prev = v
	ds.i++
	return v, nil
}

// nextFast is a no-error variant of next. The caller must ensure the stream
// has at least one remaining value.
func (ds *deltaStream) nextFast() byte {
	if ds.i == 0 {
		ds.i = 1
		return ds.prev
	}
	d := int8(ds.packed[ds.i])
	ds.prev = decodeDelta8(ds.prev, d)
	ds.i++
	return ds.prev
}
