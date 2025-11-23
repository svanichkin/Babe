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
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
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
var encQuality int = 100

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

// BitWriter writes bits to an io.ByteWriter (e.g., *bytes.Buffer).
type BitWriter struct {
	w    io.ByteWriter
	byte byte
	n    uint8 // number of bits written (0..8)
}

func NewBitWriter(w io.ByteWriter) *BitWriter {
	return &BitWriter{w: w}
}

// WriteBit writes a single bit (msb-first in byte).
func (bw *BitWriter) WriteBit(bit bool) error {
	bw.byte <<= 1
	if bit {
		bw.byte |= 1
	}
	bw.n++
	if bw.n == 8 {
		if err := bw.w.WriteByte(bw.byte); err != nil {
			return err
		}
		bw.byte = 0
		bw.n = 0
	}
	return nil
}

// Flush writes any remaining bits, left-padded with zeros.
func (bw *BitWriter) Flush() error {
	if bw.n > 0 {
		bw.byte <<= 8 - bw.n
		if err := bw.w.WriteByte(bw.byte); err != nil {
			return err
		}
		bw.byte = 0
		bw.n = 0
	}
	return nil
}

// BitReader reads bits from a byte slice (msb-first in each byte).
type BitReader struct {
	data []byte
	idx  int
	bit  uint8 // bit position in current byte (0..7), msb-first
}

func NewBitReader(data []byte) *BitReader {
	return &BitReader{data: data, idx: 0, bit: 0}
}

// ReadBit returns the next bit, or error if out of data.
func (br *BitReader) ReadBit() (bool, error) {
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

// blockMetaCh describes a single block for one channel (Y, Cb, or Cr).
// Geometry is implicit from the traversal order, but we keep a flag to distinguish
// macroBlocks from smallBlocks so we can encode FG/BG differently.
type blockMetaCh struct {
	fg, bg uint8
	isBig  bool // true for macroBlock-sized blocks, false for smallBlock
}

// channel IDs for Y, Cb, Cr.
const (
	chY  = 0
	chCb = 1
	chCr = 2
)

// extractYCbCrPlanes converts an image.Image into three planar Y, Cb, Cr slices.
// Each plane has size w*h and is indexed as plane[y*w + x] with 0 <= x < w and 0 <= y < h.
func extractYCbCrPlanes(img image.Image) ([]uint8, []uint8, []uint8, int, int) {
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()

	yPlane := make([]uint8, w*h)
	cbPlane := make([]uint8, w*h)
	crPlane := make([]uint8, w*h)

	// Parallelize over scanlines to speed up the expensive At/RGBA work.
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
		go func(yStart, yEnd int) {
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
		}(y0, y1)
	}
	wg.Wait()

	return yPlane, cbPlane, crPlane, w, h
}

// canUseBigBlockChannel decides whether a macroBlock region can be encoded as a single block
// for the given channel plane. It uses a quality-dependent spread threshold: lower quality
// allows larger spread (more macroBlocks), higher quality reduces spread (more small blocks).
func canUseBigBlockChannel(plane []uint8, stride, height, x0, y0 int) bool {
	if x0+macroBlock > stride || y0+macroBlock > height {
		return false
	}

	var minV, maxV int32
	first := true
	for yy := 0; yy < macroBlock; yy++ {
		for xx := 0; xx < macroBlock; xx++ {
			idx := (y0+yy)*stride + (x0 + xx)
			if idx < 0 || idx >= len(plane) {
				return false
			}
			v := int32(plane[idx])
			if first {
				minV, maxV = v, v
				first = false
			} else {
				if v < minV {
					minV = v
				}
				if v > maxV {
					maxV = v
				}
			}
		}
	}

	// quality-dependent spread:
	// in each quality band (0–19, 20–39, 40–59, 60–79, 80–100) that we use
	// for smallBlock/macroBlock selection, the allowed spread changes linearly
	// from 64 (bottom of the band) down to 8 (top of the band).
	q := encQuality
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

	return (maxV - minV) < spread
}

// encodeBlockPlane encodes a single block for one planar channel:
// it computes a mean-based threshold, writes the FG/BG pattern bits (if pw != nil),
// and computes FG/BG levels.
func encodeBlockPlane(plane []uint8, stride, x0, y0, bw, bh int, pw *BitWriter) (uint8, uint8, error) {
	total := bw * bh
	if total <= 0 {
		return 0, 0, fmt.Errorf("invalid block size")
	}

	// first pass: compute average for the chosen channel
	var sum uint64
	for yy := 0; yy < bh; yy++ {
		for xx := 0; xx < bw; xx++ {
			idx := (y0+yy)*stride + (x0 + xx)
			if idx >= len(plane) {
				return 0, 0, fmt.Errorf("encodeBlockPlane: index out of range")
			}
			val := plane[idx]
			sum += uint64(val)
		}
	}
	thr := int32(sum / uint64(total))

	// second pass: emit bits and accumulate FG/BG means
	var fgSum, bgSum uint64
	var fgCnt, bgCnt uint32

	emitBit := func(isFg bool) error {
		if pw != nil {
			if err := pw.WriteBit(isFg); err != nil {
				return err
			}
		}
		return nil
	}

	for yy := 0; yy < bh; yy++ {
		for xx := 0; xx < bw; xx++ {
			idx := (y0+yy)*stride + (x0 + xx)
			if idx >= len(plane) {
				return 0, 0, fmt.Errorf("encodeBlockPlane: index out of range")
			}
			v := int32(plane[idx])
			isFg := v >= thr
			if err := emitBit(isFg); err != nil {
				return 0, 0, err
			}
			if isFg {
				fgSum += uint64(v)
				fgCnt++
			} else {
				bgSum += uint64(v)
				bgCnt++
			}
		}
	}


	// handle degenerate cases: if one of the groups is empty, fall back to the global mean
	avg := uint8(sum / uint64(total))
	var fg, bg uint8
	if fgCnt == 0 || bgCnt == 0 {
		fg, bg = avg, avg
	} else {
		fg = uint8(fgSum / uint64(fgCnt))
		bg = uint8(bgSum / uint64(bgCnt))
	}

	return fg, bg, nil
}

// encodeChannel builds one complete stream for a single planar channel:
// its own block list, size stream, pattern stream, and per-block FG/BG levels.
func encodeChannel(plane []uint8, stride, w4, h4, fullW, fullH int, useMacro bool) ([]blockMetaCh, []byte, []byte, []byte, []uint8, []uint8, error) {
	// macro-block decision bits (only for main fullW x fullH area)
	var macroUseBig []bool
	var blocks []blockMetaCh

	var sizeBuf bytes.Buffer
	sizeW := NewBitWriter(&sizeBuf)
	var patternBuf bytes.Buffer
	patternW := NewBitWriter(&patternBuf)

	var typeBuf bytes.Buffer
	typeW := NewBitWriter(&typeBuf)

	var fgVals []uint8
	var bgVals []uint8

	height := h4

	// main macroBlock x macroBlock area
	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			if useMacro && canUseBigBlockChannel(plane, stride, height, mx, my) {
				fg, bg, err := encodeBlockPlane(plane, stride, mx, my, macroBlock, macroBlock, patternW)
				if err != nil {
					return nil, nil, nil, nil, nil, nil, err
				}
				fgVals = append(fgVals, fg)
				bgVals = append(bgVals, bg)
				if err := typeW.WriteBit(true); err != nil { // always pattern
					return nil, nil, nil, nil, nil, nil, err
				}
				blocks = append(blocks, blockMetaCh{fg: fg, bg: bg, isBig: true})
				macroUseBig = append(macroUseBig, true)
			} else {
				// grid of smallBlock x smallBlock blocks covering the macroBlock area
				for by := 0; by < macroBlock; by += smallBlock {
					for bx := 0; bx < macroBlock; bx += smallBlock {
						xx, yy := mx+bx, my+by
						// for smallBlock > 1 we also emit pattern bits; for smallBlock == 1 we skip pattern bits
						var pw *BitWriter
						isPattern := false
						if smallBlock > 1 {
							pw = patternW
							isPattern = true
						}
						fg, bg, err := encodeBlockPlane(plane, stride, xx, yy, smallBlock, smallBlock, pw)
						if err != nil {
							return nil, nil, nil, nil, nil, nil, err
						}
						fgVals = append(fgVals, fg)
						bgVals = append(bgVals, bg)
						if err := typeW.WriteBit(isPattern); err != nil {
							return nil, nil, nil, nil, nil, nil, err
						}
						blocks = append(blocks, blockMetaCh{fg: fg, bg: bg, isBig: false})
					}
				}
				macroUseBig = append(macroUseBig, false)
			}
		}
	}

	// right stripe: small blocks only
	for my := 0; my < fullH; my += smallBlock {
		for mx := fullW; mx < w4; mx += smallBlock {
			var pw *BitWriter
			isPattern := false
			if smallBlock > 1 {
				pw = patternW
				isPattern = true
			}
			fg, bg, err := encodeBlockPlane(plane, stride, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return nil, nil, nil, nil, nil, nil, err
			}
			fgVals = append(fgVals, fg)
			bgVals = append(bgVals, bg)
			if err := typeW.WriteBit(isPattern); err != nil {
				return nil, nil, nil, nil, nil, nil, err
			}
			blocks = append(blocks, blockMetaCh{fg: fg, bg: bg, isBig: false})
		}
	}
	// bottom stripe: small blocks only (including bottom-right corner)
	for my := fullH; my < h4; my += smallBlock {
		for mx := 0; mx < w4; mx += smallBlock {
			var pw *BitWriter
			isPattern := false
			if smallBlock > 1 {
				pw = patternW
				isPattern = true
			}
			fg, bg, err := encodeBlockPlane(plane, stride, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return nil, nil, nil, nil, nil, nil, err
			}
			fgVals = append(fgVals, fg)
			bgVals = append(bgVals, bg)
			if err := typeW.WriteBit(isPattern); err != nil {
				return nil, nil, nil, nil, nil, nil, err
			}
			blocks = append(blocks, blockMetaCh{fg: fg, bg: bg, isBig: false})
		}
	}

	// write macroUseBig bits for the main area into the size stream
	for _, useBig := range macroUseBig {
		if err := sizeW.WriteBit(useBig); err != nil {
			return nil, nil, nil, nil, nil, nil, err
		}
	}
	if err := sizeW.Flush(); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := typeW.Flush(); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	if err := patternW.Flush(); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	return blocks, sizeBuf.Bytes(), typeBuf.Bytes(), patternBuf.Bytes(), fgVals, bgVals, nil
}

// Encode runs the BABE encoder with three fully independent channel streams (Y, Cb, Cr).
// Each channel has its own block list, size stream, pattern stream, and FG/BG levels.
func Encode(img image.Image, quality int) ([]byte, error) {
	if err := setBlocksForQuality(quality); err != nil {
		return nil, err
	}

	yPlane, cbPlane, crPlane, w, h := extractYCbCrPlanes(img)

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
	if err := binary.Write(bw, binary.BigEndian, uint16(smallBlock)); err != nil {
		return nil, err
	}
	if err := binary.Write(bw, binary.BigEndian, uint16(macroBlock)); err != nil {
		return nil, err
	}
	if err := binary.Write(bw, binary.BigEndian, uint32(w)); err != nil {
		return nil, err
	}
	if err := binary.Write(bw, binary.BigEndian, uint32(h)); err != nil {
		return nil, err
	}

	// --- Encode three independent channels: Y, Cb, Cr ---
	type channelResult struct {
		blocks       []blockMetaCh
		sizeBytes    []byte
		typeBytes    []byte
		patternBytes []byte
		fgVals       []uint8
		bgVals       []uint8
		err          error
	}

	var (
		results [3]channelResult
		wg      sync.WaitGroup
	)

	channelIDs := []int{chY, chCb, chCr}
	planes := [][]uint8{yPlane, cbPlane, crPlane}

	for i := range planes {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			blocks, sizeBytes, typeBytes, patternBytes, fgVals, bgVals, err := encodeChannel(planes[i], w, w4, h4, fullW, fullH, useMacro)
			results[i] = channelResult{
				blocks:       blocks,
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

	// Write channels in fixed order: Y, Cb, Cr.
	// For experimentation: smallBlocks store only FG (BG is implicit = 0),
	// macroBlocks store both FG and BG like before.
	for i := 0; i < 3; i++ {
		res := results[i]
		if res.err != nil {
			return nil, res.err
		}
		blockCount := uint32(len(res.blocks))
		// number of blocks for this channel
		if err := binary.Write(bw, binary.BigEndian, blockCount); err != nil {
			return nil, err
		}
		// write size stream for this channel
		if err := binary.Write(bw, binary.BigEndian, uint32(len(res.sizeBytes))); err != nil {
			return nil, err
		}
		if _, err := bw.Write(res.sizeBytes); err != nil {
			return nil, err
		}
		// write type stream for this channel
		if err := binary.Write(bw, binary.BigEndian, uint32(len(res.typeBytes))); err != nil {
			return nil, err
		}
		if _, err := bw.Write(res.typeBytes); err != nil {
			return nil, err
		}
		// write pattern stream for this channel
		if err := binary.Write(bw, binary.BigEndian, uint32(len(res.patternBytes))); err != nil {
			return nil, err
		}
		if _, err := bw.Write(res.patternBytes); err != nil {
			return nil, err
		}
		// write FG array:
		// mode 0 = delta-coded via DeltaPackBytes (used for Y),
		// mode 1 = packed with PackBytes (used for Cb/Cr).
		chID := channelIDs[i]
		if chID == chY {
			// Y channel: delta scheme via DeltaPackBytes
			if err := bw.WriteByte(0); err != nil {
				return nil, err
			}
			packedFG, err := DeltaPackBytes(res.fgVals)
			if err != nil {
				return nil, err
			}
			fgLen := uint32(len(packedFG))
			if err := binary.Write(bw, binary.BigEndian, fgLen); err != nil {
				return nil, err
			}
			if _, err := bw.Write(packedFG); err != nil {
				return nil, err
			}
		} else {
			// Cb/Cr: pack with PackBytes
			if err := bw.WriteByte(1); err != nil {
				return nil, err
			}
			packedFG, err := DeltaPackBytes(res.fgVals)
			if err != nil {
				return nil, err
			}
			fgLen := uint32(len(packedFG))
			if err := binary.Write(bw, binary.BigEndian, fgLen); err != nil {
				return nil, err
			}
			if _, err := bw.Write(packedFG); err != nil {
				return nil, err
			}
		}

		// write BG array:
		if chID == chY {
			// Y channel: delta scheme via DeltaPackBytes
			if err := bw.WriteByte(0); err != nil {
				return nil, err
			}
			packedBG, err := DeltaPackBytes(res.bgVals)
			if err != nil {
				return nil, err
			}
			bgLen := uint32(len(packedBG))
			if err := binary.Write(bw, binary.BigEndian, bgLen); err != nil {
				return nil, err
			}
			if _, err := bw.Write(packedBG); err != nil {
				return nil, err
			}
		} else {
			// Cb/Cr: pack with PackBytes
			if err := bw.WriteByte(1); err != nil {
				return nil, err
			}
			packedBG, err := DeltaPackBytes(res.bgVals)
			if err != nil {
				return nil, err
			}
			bgLen := uint32(len(packedBG))
			if err := binary.Write(bw, binary.BigEndian, bgLen); err != nil {
				return nil, err
			}
			if _, err := bw.Write(packedBG); err != nil {
				return nil, err
			}
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
func drawBlockPlane(plane []uint8, stride int, x0, y0, bw, bh int, br *BitReader, fg, bg uint8) error {
	for yy := 0; yy < bh; yy++ {
		for xx := 0; xx < bw; xx++ {
			bit, err := br.ReadBit()
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

// readChannelSegment reads a single channel stream (as written by Encode)
// from the shared reader into a standalone byte slice. The layout of this
// slice matches what decodeChannel expects, so decodeChannel can run on
// an independent reader in parallel for each channel.
func readChannelSegment(br *bufio.Reader) ([]byte, error) {
	var buf bytes.Buffer
	var tmp4 [4]byte

	// Helper to read a uint32, write it into buf, and return the value.
	readU32 := func() (uint32, error) {
		if _, err := io.ReadFull(br, tmp4[:]); err != nil {
			return 0, err
		}
		if _, err := buf.Write(tmp4[:]); err != nil {
			return 0, err
		}
		return binary.BigEndian.Uint32(tmp4[:]), nil
	}
	// Helper to read n bytes from br, write to buf.
	readBytes := func(n uint32) error {
		if n == 0 {
			return nil
		}
		b := make([]byte, n)
		if _, err := io.ReadFull(br, b); err != nil {
			return err
		}
		_, err := buf.Write(b)
		return err
	}

	// blockCount (4 bytes)
	if _, err := readU32(); err != nil {
		return nil, err
	}
	// sizeStreamLen (4 bytes)
	sizeStreamLen, err := readU32()
	if err != nil {
		return nil, err
	}
	if err := readBytes(sizeStreamLen); err != nil {
		return nil, err
	}
	// typeStreamLen (4 bytes)
	typeStreamLen, err := readU32()
	if err != nil {
		return nil, err
	}
	if err := readBytes(typeStreamLen); err != nil {
		return nil, err
	}
	// patternLen (4 bytes)
	patternLen, err := readU32()
	if err != nil {
		return nil, err
	}
	if err := readBytes(patternLen); err != nil {
		return nil, err
	}
	// FG: mode (1 byte)
	b, err := br.ReadByte()
	if err != nil {
		return nil, err
	}
	if err := buf.WriteByte(b); err != nil {
		return nil, err
	}
	// FG packed length (4 bytes)
	fgPackedLen, err := readU32()
	if err != nil {
		return nil, err
	}
	if err := readBytes(fgPackedLen); err != nil {
		return nil, err
	}
	// BG: mode (1 byte)
	b, err = br.ReadByte()
	if err != nil {
		return nil, err
	}
	if err := buf.WriteByte(b); err != nil {
		return nil, err
	}
	// BG packed length (4 bytes)
	bgPackedLen, err := readU32()
	if err != nil {
		return nil, err
	}
	if err := readBytes(bgPackedLen); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// decodeChannel decodes one channel stream into a planar buffer of size imgW x imgH.
func decodeChannel(br *bufio.Reader, imgW, imgH int) ([]uint8, error) {
	var blockCount uint32
	if err := binary.Read(br, binary.BigEndian, &blockCount); err != nil {
		return nil, err
	}

	var sizeStreamLen uint32
	if err := binary.Read(br, binary.BigEndian, &sizeStreamLen); err != nil {
		return nil, err
	}
	sizeBytes := make([]byte, sizeStreamLen)
	if _, err := io.ReadFull(br, sizeBytes); err != nil {
		return nil, err
	}
	sizeBR := NewBitReader(sizeBytes)

	var typeStreamLen uint32
	if err := binary.Read(br, binary.BigEndian, &typeStreamLen); err != nil {
		return nil, err
	}
	typeBytes := make([]byte, typeStreamLen)
	if _, err := io.ReadFull(br, typeBytes); err != nil {
		return nil, err
	}
	typeBR := NewBitReader(typeBytes)

	var patternLen uint32
	if err := binary.Read(br, binary.BigEndian, &patternLen); err != nil {
		return nil, err
	}
	patternBytes := make([]byte, patternLen)
	if _, err := io.ReadFull(br, patternBytes); err != nil {
		return nil, err
	}
	patternBR := NewBitReader(patternBytes)

	// FG: read mode and decode accordingly
	modeFG, err := br.ReadByte()
	if err != nil {
		return nil, err
	}
	var fgVals []uint8
	var fgCount32 uint32
	if modeFG == 0 {
		// delta-coded via DeltaPackBytes/DeltaUnpackBytes
		var packedLen uint32
		if err := binary.Read(br, binary.BigEndian, &packedLen); err != nil {
			return nil, err
		}
		packed := make([]byte, packedLen)
		if _, err := io.ReadFull(br, packed); err != nil {
			return nil, err
		}
		fgVals, err = DeltaUnpackBytes(packed)
		if err != nil {
			return nil, err
		}
		fgCount32 = uint32(len(fgVals))
	} else {
		// packed with PackBytes
		var packedLen uint32
		if err := binary.Read(br, binary.BigEndian, &packedLen); err != nil {
			return nil, err
		}
		packed := make([]byte, packedLen)
		if _, err := io.ReadFull(br, packed); err != nil {
			return nil, err
		}
		fgVals, err = DeltaUnpackBytes(packed)
		if err != nil {
			return nil, err
		}
		fgCount32 = uint32(len(fgVals))
	}

	// BG: read mode and decode accordingly
	modeBG, err := br.ReadByte()
	if err != nil {
		return nil, err
	}
	var bgVals []uint8
	var bgCount32 uint32
	if modeBG == 0 {
		// delta-coded via DeltaPackBytes/DeltaUnpackBytes
		var packedLen uint32
		if err := binary.Read(br, binary.BigEndian, &packedLen); err != nil {
			return nil, err
		}
		packed := make([]byte, packedLen)
		if _, err := io.ReadFull(br, packed); err != nil {
			return nil, err
		}
		bgVals, err = DeltaUnpackBytes(packed)
		if err != nil {
			return nil, err
		}
		bgCount32 = uint32(len(bgVals))
	} else {
		// packed with PackBytes
		var packedLen uint32
		if err := binary.Read(br, binary.BigEndian, &packedLen); err != nil {
			return nil, err
		}
		packed := make([]byte, packedLen)
		if _, err := io.ReadFull(br, packed); err != nil {
			return nil, err
		}
		bgVals, err = DeltaUnpackBytes(packed)
		if err != nil {
			return nil, err
		}
		bgCount32 = uint32(len(bgVals))
	}

	if fgCount32 != blockCount || bgCount32 != blockCount {
		return nil, fmt.Errorf("color count mismatch: fg=%d bg=%d blocks=%d", fgCount32, bgCount32, blockCount)
	}

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
	for i := 0; i < macroGroupCount; i++ {
		bit, err := sizeBR.ReadBit()
		if err != nil {
			return nil, err
		}
		macroBig[i] = bit
	}

	blockIndex := 0
	macroIndex := 0
	fgIndex := 0
	bgIndex := 0

	// main macroBlock x macroBlock area
	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			if blockIndex >= int(blockCount) {
				return nil, fmt.Errorf("unexpected end of blocks in main area")
			}
			if useMacro && macroBig[macroIndex] {
				// macroBlock: read one type bit, FG and BG from arrays, use full pattern
				bitType, err := typeBR.ReadBit()
				if err != nil {
					return nil, err
				}
				_ = bitType // always true for macro blocks
				fg := fgVals[fgIndex]
				bg := bgVals[bgIndex]
				fgIndex++
				bgIndex++
				if err := drawBlockPlane(plane, imgW, mx, my, macroBlock, macroBlock, patternBR, fg, bg); err != nil {
					return nil, err
				}
				blockIndex++
			} else {
				// grid of smallBlock x smallBlock blocks covering the macroBlock area
				for by := 0; by < macroBlock; by += smallBlock {
					for bx := 0; bx < macroBlock; bx += smallBlock {
						if blockIndex >= int(blockCount) {
							return nil, fmt.Errorf("unexpected end of blocks in macro grid")
						}
						bitType, err := typeBR.ReadBit()
						if err != nil {
							return nil, err
						}
						fg := fgVals[fgIndex]
						bg := bgVals[bgIndex]
						fgIndex++
						bgIndex++
						if bitType {
							if err := drawBlockPlane(plane, imgW, mx+bx, my+by, smallBlock, smallBlock, patternBR, fg, bg); err != nil {
								return nil, err
							}
						} else {
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
			bitType, err := typeBR.ReadBit()
			if err != nil {
				return nil, err
			}
			fg := fgVals[fgIndex]
			bg := bgVals[bgIndex]
			fgIndex++
			bgIndex++
			if bitType {
				if err := drawBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, patternBR, fg, bg); err != nil {
					return nil, err
				}
			} else {
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
			bitType, err := typeBR.ReadBit()
			if err != nil {
				return nil, err
			}
			fg := fgVals[fgIndex]
			bg := bgVals[bgIndex]
			fgIndex++
			bgIndex++
			if bitType {
				if err := drawBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, patternBR, fg, bg); err != nil {
					return nil, err
				}
			} else {
				if err := fillBlockPlane(plane, imgW, mx, my, smallBlock, smallBlock, fg); err != nil {
					return nil, err
				}
			}
			blockIndex++
		}
	}
	if fgIndex != int(blockCount) || bgIndex != int(blockCount) {
		return nil, fmt.Errorf("color stream mismatch: used fg=%d bg=%d blocks=%d", fgIndex, bgIndex, blockCount)
	}
	if blockIndex != int(blockCount) {
		return nil, fmt.Errorf("block count mismatch: used %d of %d", blockIndex, blockCount)
	}

	return plane, nil
}

// Decode reads a BABE-compressed stream (Zstd + three independent Y/Cb/Cr streams)
// and returns an image.Image.
func Decode(compData []byte) (image.Image, error) {
	payload, err := decompressZstd(compData)
	if err != nil {
		return nil, fmt.Errorf("zstd decode: %w", err)
	}

	br := bufio.NewReader(bytes.NewReader(payload))

	// magic
	header := make([]byte, len(codec))
	if _, err := io.ReadFull(br, header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(header) != codec {
		return nil, fmt.Errorf("bad magic: %q", string(header))
	}

	var bwSize, bhSize uint16
	var imgW32, imgH32 uint32

	if err := binary.Read(br, binary.BigEndian, &bwSize); err != nil {
		return nil, err
	}
	if err := binary.Read(br, binary.BigEndian, &bhSize); err != nil {
		return nil, err
	}
	if bwSize == 0 || bhSize == 0 {
		return nil, fmt.Errorf("invalid block sizes in header: %dx%d", bwSize, bhSize)
	}
	smallBlock = int(bwSize)
	macroBlock = int(bhSize)

	if err := binary.Read(br, binary.BigEndian, &imgW32); err != nil {
		return nil, err
	}
	if err := binary.Read(br, binary.BigEndian, &imgH32); err != nil {
		return nil, err
	}
	imgW := int(imgW32)
	imgH := int(imgH32)

	if macroBlock < smallBlock || macroBlock%smallBlock != 0 {
		return nil, fmt.Errorf("macroBlock (%d) must be >= smallBlock (%d) and a multiple of it",
			macroBlock, smallBlock)
	}

	// read three channel segments (Y, Cb, Cr) sequentially,
	// then decode each one in its own goroutine.
	ySeg, err := readChannelSegment(br)
	if err != nil {
		return nil, err
	}
	cbSeg, err := readChannelSegment(br)
	if err != nil {
		return nil, err
	}
	crSeg, err := readChannelSegment(br)
	if err != nil {
		return nil, err
	}

	type chResult struct {
		plane []uint8
		err   error
	}

	var (
		resY  chResult
		resCb chResult
		resCr chResult
		wg    sync.WaitGroup
	)

	wg.Add(3)
	go func() {
		defer wg.Done()
		r := bufio.NewReader(bytes.NewReader(ySeg))
		resY.plane, resY.err = decodeChannel(r, imgW, imgH)
	}()
	go func() {
		defer wg.Done()
		r := bufio.NewReader(bytes.NewReader(cbSeg))
		resCb.plane, resCb.err = decodeChannel(r, imgW, imgH)
	}()
	go func() {
		defer wg.Done()
		r := bufio.NewReader(bytes.NewReader(crSeg))
		resCr.plane, resCr.err = decodeChannel(r, imgW, imgH)
	}()
	wg.Wait()

	if resY.err != nil {
		return nil, resY.err
	}
	if resCb.err != nil {
		return nil, resCb.err
	}
	if resCr.err != nil {
		return nil, resCr.err
	}

	yPlane := resY.plane
	cbPlane := resCb.plane
	crPlane := resCr.plane

	// combine into RGBA (fast path: write directly into Pix)
	dst := image.NewRGBA(image.Rect(0, 0, imgW, imgH))
	pix := dst.Pix
	stride := dst.Stride
	for y := 0; y < imgH; y++ {
		rowOff := y * stride
		baseIdx := y * imgW
		for x := 0; x < imgW; x++ {
			idx := baseIdx + x
			Y := float64(yPlane[idx])
			Cb := float64(cbPlane[idx]) - 128.0
			Cr := float64(crPlane[idx]) - 128.0
			R := Y + 1.402*Cr
			G := Y - 0.344136*Cb - 0.714136*Cr
			B := Y + 1.772*Cb

			// clamp to [0,255]
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

			o := rowOff + x*4
			pix[o+0] = uint8(R)
			pix[o+1] = uint8(G)
			pix[o+2] = uint8(B)
			pix[o+3] = 255
		}
	}

	// Post-process: gradient smoothing at junctions + light deblocking.
	smoothed := smoothBlocks(dst)

	return smoothed, nil
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

	maxRadius := smallBlock
	if maxRadius > 3 {
		maxRadius = 3
	}
	const junctionLumaSpread int32 = 32
	const lumaSimThreshold int32 = 8

	minX := b.Min.X
	minY := b.Min.Y
	stride := img.Stride
	pix := img.Pix

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

	for y := b.Min.Y + smallBlock; y < b.Max.Y; y += smallBlock {
		for x := b.Min.X + smallBlock; x < b.Max.X; x += smallBlock {
			// four corners around the junction
			idx00 := (y-1-minY)*stride + (x-1-minX)*4 // top-left
			idx10 := (y-1-minY)*stride + (x-minX)*4   // top-right
			idx01 := (y-minY)*stride + (x-1-minX)*4   // bottom-left
			idx11 := (y-minY)*stride + (x-minX)*4     // bottom-right

			r00, g00, b00, a00 := pix[idx00+0], pix[idx00+1], pix[idx00+2], pix[idx00+3]
			r10, g10, b10, a10 := pix[idx10+0], pix[idx10+1], pix[idx10+2], pix[idx10+3]
			r01, g01, b01, a01 := pix[idx01+0], pix[idx01+1], pix[idx01+2], pix[idx01+3]
			r11, g11, b11, a11 := pix[idx11+0], pix[idx11+1], pix[idx11+2], pix[idx11+3]

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
				rUp, gUp, bUp, aUp := pix[idxUp+0], pix[idxUp+1], pix[idxUp+2], pix[idxUp+3]
				rDown, gDown, bDown, aDown := pix[idxDown+0], pix[idxDown+1], pix[idxDown+2], pix[idxDown+3]
				if !similar(rUp, gUp, bUp, aUp, r00, g00, b00, a00) || !similar(rDown, gDown, bDown, aDown, r01, g01, b01, a01) {
					break
				}
				L = i
			}

			R := 0
			for i := 1; i <= maxRadius && x-1+i < b.Max.X; i++ {
				idxUp := (y-1-minY)*stride + (x-1+i-minX)*4
				idxDown := (y-minY)*stride + (x-1+i-minX)*4
				rUp, gUp, bUp, aUp := pix[idxUp+0], pix[idxUp+1], pix[idxUp+2], pix[idxUp+3]
				rDown, gDown, bDown, aDown := pix[idxDown+0], pix[idxDown+1], pix[idxDown+2], pix[idxDown+3]
				if !similar(rUp, gUp, bUp, aUp, r10, g10, b10, a10) || !similar(rDown, gDown, bDown, aDown, r11, g11, b11, a11) {
					break
				}
				R = i
			}

			U := 0
			for i := 1; i <= maxRadius && y-i >= b.Min.Y; i++ {
				idxLeft := (y-i-minY)*stride + (x-1-minX)*4
				idxRight := (y-i-minY)*stride + (x-minX)*4
				rLeft, gLeft, bLeft, aLeft := pix[idxLeft+0], pix[idxLeft+1], pix[idxLeft+2], pix[idxLeft+3]
				rRight, gRight, bRight, aRight := pix[idxRight+0], pix[idxRight+1], pix[idxRight+2], pix[idxRight+3]
				if !similar(rLeft, gLeft, bLeft, aLeft, r00, g00, b00, a00) || !similar(rRight, gRight, bRight, aRight, r10, g10, b10, a10) {
					break
				}
				U = i
			}

			D := 0
			for i := 1; i <= maxRadius && y-1+i < b.Max.Y; i++ {
				idxLeft := (y-1+i-minY)*stride + (x-1-minX)*4
				idxRight := (y-1+i-minY)*stride + (x-minX)*4
				rLeft, gLeft, bLeft, aLeft := pix[idxLeft+0], pix[idxLeft+1], pix[idxLeft+2], pix[idxLeft+3]
				rRight, gRight, bRight, aRight := pix[idxRight+0], pix[idxRight+1], pix[idxRight+2], pix[idxRight+3]
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
					pix[idx+0] = r
					pix[idx+1] = g
					pix[idx+2] = bc
					pix[idx+3] = a
				}
			}
		}
	}

	return img
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
	for x := b.Min.X + smallBlock; x < b.Max.X; x += smallBlock {
		xL := x - 1
		for y := b.Min.Y; y < b.Max.Y; y++ {
			rowOff := (y - minY) * stride
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

	// horizontal block boundaries: y = k * smallBlock
	for y := b.Min.Y + smallBlock; y < b.Max.Y; y += smallBlock {
		yT := y - 1
		for x := b.Min.X; x < b.Max.X; x++ {
			colOff := (x - minX) * 4
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

	return src
}

// smoothBlocks runs both junction smoothing and light deblocking in a fixed order.
// This keeps the post-process logic in one place for Decode().
func smoothBlocks(src *image.RGBA) *image.RGBA {
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
	R := float64(r)
	G := float64(g)
	B := float64(b)

	Y := int16(0.299*R + 0.587*G + 0.114*B)
	Cb := int16(-0.169*R - 0.331*G + 0.5*B + 128)
	Cr := int16(0.5*R - 0.418*G - 0.081*B + 128)

	return yuv{Y, Cb, Cr}
}

func yCbCrToRGB(v yuv) [3]uint8 {
	Y := float64(v.Y)
	Cb := float64(v.Cb) - 128
	Cr := float64(v.Cr) - 128

	R := Y + 1.402*Cr
	G := Y - 0.344136*Cb - 0.714136*Cr
	B := Y + 1.772*Cb

	clip := func(x float64) uint8 {
		if x < 0 {
			return 0
		}
		if x > 255 {
			return 255
		}
		return uint8(x)
	}

	return [3]uint8{clip(R), clip(G), clip(B)}
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

func compressZstd(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}
	var buf bytes.Buffer
	enc, err := zstd.NewWriter(&buf)
	if err != nil {
		return nil, err
	}
	if _, err := enc.Write(data); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decompressZstd(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}
	dec, err := zstd.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer dec.Close()

	var out bytes.Buffer
	if _, err := out.ReadFrom(dec); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

func luma(c color.Color) int32 {
	r, g, b, _ := c.RGBA()
	rr := int32(r >> 8)
	gg := int32(g >> 8)
	bb := int32(b >> 8)

	// Integer approximation of ITU-R BT.601:
	// Y ≈ 0.299 R + 0.587 G + 0.114 B
	return (299*rr + 587*gg + 114*bb) / 1000
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

// DeltaPackBytes encodes a byte slice using circular delta coding on 0..255.
// Layout:
//
//	[0..3]  = totalLen (uint32, BigEndian) — original length
//	[4]     = first byte (if totalLen > 0)
//	[5..]   = (totalLen-1) delta bytes (encodeDelta8(prev, curr))
func DeltaPackBytes(src []byte) ([]byte, error) {
	n := len(src)

	// Always allocate header (4 bytes) + n payload bytes.
	// For n == 0, this yields a 4-byte slice with just the length header.
	out := make([]byte, 4+n)

	// total length
	binary.BigEndian.PutUint32(out[:4], uint32(n))
	if n == 0 {
		return out, nil
	}

	// first byte as-is
	out[4] = src[0]

	// remaining bytes as deltas
	prev := src[0]
	for i := 1; i < n; i++ {
		d := encodeDelta8(prev, src[i])
		out[4+i] = byte(d)
		prev = src[i]
	}

	return out, nil
}

// DeltaUnpackBytes decodes a slice produced by DeltaPackBytes.
func DeltaUnpackBytes(packed []byte) ([]byte, error) {
	if len(packed) < 4 {
		return nil, fmt.Errorf("delta packed data too short")
	}

	totalLen := binary.BigEndian.Uint32(packed[0:4])
	if totalLen == 0 {
		return []byte{}, nil
	}

	// need at least one byte for the first value
	if len(packed) < 4+1 {
		return nil, fmt.Errorf("delta packed data truncated (no first byte)")
	}

	dst := make([]byte, totalLen)
	pos := 4

	// first byte
	first := packed[pos]
	dst[0] = first
	pos++

	// remaining deltas
	expectedDeltaCount := int(totalLen) - 1
	if len(packed) < pos+expectedDeltaCount {
		return nil, fmt.Errorf("delta packed data truncated (missing deltas)")
	}

	prev := first
	for i := 1; i < int(totalLen); i++ {
		d := int8(packed[pos])
		pos++
		val := decodeDelta8(prev, d)
		dst[i] = val
		prev = val
	}

	return dst, nil
}
