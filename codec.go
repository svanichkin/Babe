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

// bitWriter writes bits to an io.ByteWriter (e.g., *bytes.Buffer).
type bitWriter struct {
	w    io.ByteWriter
	byte byte
	n    uint8 // number of bits written (0..8)
}

func newBitWriter(w io.ByteWriter) *bitWriter {
	return &bitWriter{w: w}
}

// writeBit writes a single bit (msb-first in byte).
func (bw *bitWriter) writeBit(bit bool) error {
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

// flush writes any remaining bits, left-padded with zeros.
func (bw *bitWriter) flush() error {
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

// bitReader reads bits from a byte slice (msb-first in each byte).
type bitReader struct {
	data []byte
	idx  int
	bit  uint8 // bit position in current byte (0..7), msb-first
}

func newBitReader(data []byte) *bitReader {
	return &bitReader{data: data, idx: 0, bit: 0}
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

	switch src := img.(type) {
	case *image.RGBA:
		extractYCbCrFromRGBA(src, yPlane, cbPlane, crPlane, w, h)
	case *image.NRGBA:
		extractYCbCrFromNRGBA(src, yPlane, cbPlane, crPlane, w, h)
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
	}

	return yPlane, cbPlane, crPlane, w, h
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
		go func(yStart, yEnd int) {
			defer wg.Done()
			for y := yStart; y < yEnd; y++ {
				baseIdx := y * w
				pixRow := y * stride
				for x := 0; x < w; x++ {
					p := pixRow + x*4
					r8 := pix[p+0]
					g8 := pix[p+1]
					b8 := pix[p+2]
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
		go func(yStart, yEnd int) {
			defer wg.Done()
			for y := yStart; y < yEnd; y++ {
				baseIdx := y * w
				pixRow := y * stride
				for x := 0; x < w; x++ {
					p := pixRow + x*4
					r8 := pix[p+0]
					g8 := pix[p+1]
					b8 := pix[p+2]
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
func encodeBlockPlane(plane []uint8, stride, x0, y0, bw, bh int, pw *bitWriter) (uint8, uint8, error) {
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
			if err := pw.writeBit(isFg); err != nil {
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

	var macroUseBig []bool
	if macroGroupCount > 0 {
		macroUseBig = make([]bool, 0, macroGroupCount)
	}

	var blockCount uint32

	var sizeBuf bytes.Buffer
	sizeW := newBitWriter(&sizeBuf)
	var patternBuf bytes.Buffer
	patternW := newBitWriter(&patternBuf)

	var typeBuf bytes.Buffer
	typeW := newBitWriter(&typeBuf)

	var fgVals []uint8
	var bgVals []uint8
	if worstBlockCount > 0 {
		fgVals = make([]uint8, 0, worstBlockCount)
		bgVals = make([]uint8, 0, worstBlockCount)
	}

	height := h4

	// main macroBlock x macroBlock area
	for my := 0; my < fullH; my += macroBlock {
		for mx := 0; mx < fullW; mx += macroBlock {
			if useMacro && canUseBigBlockChannel(plane, stride, height, mx, my) {
				fg, bg, err := encodeBlockPlane(plane, stride, mx, my, macroBlock, macroBlock, patternW)
				if err != nil {
					return 0, nil, nil, nil, nil, nil, err
				}
				fgVals = append(fgVals, fg)
				bgVals = append(bgVals, bg)
				if err := typeW.writeBit(true); err != nil { // always pattern
					return 0, nil, nil, nil, nil, nil, err
				}
				blockCount++
				macroUseBig = append(macroUseBig, true)
			} else {
				// grid of smallBlock x smallBlock blocks covering the macroBlock area
				for by := 0; by < macroBlock; by += smallBlock {
					for bx := 0; bx < macroBlock; bx += smallBlock {
						xx, yy := mx+bx, my+by
						// for smallBlock > 1 we also emit pattern bits; for smallBlock == 1 we skip pattern bits
						var pw *bitWriter
						isPattern := false
						if smallBlock > 1 {
							pw = patternW
							isPattern = true
						}
						fg, bg, err := encodeBlockPlane(plane, stride, xx, yy, smallBlock, smallBlock, pw)
						if err != nil {
							return 0, nil, nil, nil, nil, nil, err
						}
						fgVals = append(fgVals, fg)
						bgVals = append(bgVals, bg)
						if err := typeW.writeBit(isPattern); err != nil {
							return 0, nil, nil, nil, nil, nil, err
						}
						blockCount++
					}
				}
				macroUseBig = append(macroUseBig, false)
			}
		}
	}

	// right stripe: small blocks only
	for my := 0; my < fullH; my += smallBlock {
		for mx := fullW; mx < w4; mx += smallBlock {
			var pw *bitWriter
			isPattern := false
			if smallBlock > 1 {
				pw = patternW
				isPattern = true
			}
			fg, bg, err := encodeBlockPlane(plane, stride, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			fgVals = append(fgVals, fg)
			bgVals = append(bgVals, bg)
			if err := typeW.writeBit(isPattern); err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			blockCount++
		}
	}
	// bottom stripe: small blocks only (including bottom-right corner)
	for my := fullH; my < h4; my += smallBlock {
		for mx := 0; mx < w4; mx += smallBlock {
			var pw *bitWriter
			isPattern := false
			if smallBlock > 1 {
				pw = patternW
				isPattern = true
			}
			fg, bg, err := encodeBlockPlane(plane, stride, mx, my, smallBlock, smallBlock, pw)
			if err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			fgVals = append(fgVals, fg)
			bgVals = append(bgVals, bg)
			if err := typeW.writeBit(isPattern); err != nil {
				return 0, nil, nil, nil, nil, nil, err
			}
			blockCount++
		}
	}

	// write macroUseBig bits for the main area into the size stream
	for _, useBig := range macroUseBig {
		if err := sizeW.writeBit(useBig); err != nil {
			return 0, nil, nil, nil, nil, nil, err
		}
	}
	if err := sizeW.flush(); err != nil {
		return 0, nil, nil, nil, nil, nil, err
	}
	if err := typeW.flush(); err != nil {
		return 0, nil, nil, nil, nil, nil, err
	}
	if err := patternW.flush(); err != nil {
		return 0, nil, nil, nil, nil, nil, err
	}

	return blockCount, sizeBuf.Bytes(), typeBuf.Bytes(), patternBuf.Bytes(), fgVals, bgVals, nil
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
	if err := binary.Write(bw, binary.BigEndian, uint16(smallBlock)); err != nil {
		return nil, err
	}
	if err := binary.Write(bw, binary.BigEndian, uint16(macroBlock)); err != nil {
		return nil, err
	}
	// channels mask: which Y/Cb/Cr planes are stored.
	if err := bw.WriteByte(channelsMask); err != nil {
		return nil, err
	}
	if err := binary.Write(bw, binary.BigEndian, uint32(w)); err != nil {
		return nil, err
	}
	if err := binary.Write(bw, binary.BigEndian, uint32(h)); err != nil {
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
	// For experimentation: smallBlocks store only FG (BG is implicit = 0),
	// macroBlocks store both FG and BG like before.
	for i, ch := range channels {
		res := results[i]
		if res.err != nil {
			return nil, res.err
		}
		blockCount := res.blockCount
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
		chID := ch.id
		if chID == chY {
			// Y channel: delta scheme via DeltaPackBytes
			if err := bw.WriteByte(0); err != nil {
				return nil, err
			}
			packedFG, err := deltaPackBytes(res.fgVals)
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
			packedFG, err := deltaPackBytes(res.fgVals)
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
			packedBG, err := deltaPackBytes(res.bgVals)
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
			packedBG, err := deltaPackBytes(res.bgVals)
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

	// FG: mode (1 byte)
	if len(data)-*pos < 1 {
		return nil, fmt.Errorf("readChannelSegment: truncated FG mode")
	}
	*pos++
	// FG packed length (4 bytes) + payload
	fgPackedLen, err := readU32("FG packedLen")
	if err != nil {
		return nil, err
	}
	if uint32(len(data)-*pos) < fgPackedLen {
		return nil, fmt.Errorf("readChannelSegment: truncated FG packed data")
	}
	*pos += int(fgPackedLen)

	// BG: mode (1 byte)
	if len(data)-*pos < 1 {
		return nil, fmt.Errorf("readChannelSegment: truncated BG mode")
	}
	*pos++
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
	if pos >= len(data) {
		return nil, fmt.Errorf("decodeChannel: truncated while reading FG mode")
	}
	modeFG := data[pos]
	_ = modeFG // currently both modes share the same delta scheme
	pos++

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
	if pos >= len(data) {
		return nil, fmt.Errorf("decodeChannel: truncated while reading BG mode")
	}
	modeBG := data[pos]
	_ = modeBG // currently both modes share the same delta scheme
	pos++

	bgPackedLen, err := readU32("BG packedLen")
	if err != nil {
		return nil, err
	}
	bgPacked, err := readSlice(bgPackedLen, "BG packed data")
	if err != nil {
		return nil, err
	}
	if blockCount > 0 && bgPackedLen < blockCount {
		return nil, fmt.Errorf("decodeChannel: BG packed data too short")
	}
	bgStream, err := newDeltaStream(bgPacked, int(blockCount))
	if err != nil {
		return nil, err
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
				_ = bitType // always true for macro blocks

				fg, err := fgStream.next()
				if err != nil {
					return nil, err
				}
				bg, err := bgStream.next()
				if err != nil {
					return nil, err
				}
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
						bitType, err := typeBR.readBit()
						if err != nil {
							return nil, err
						}
						fg, err := fgStream.next()
						if err != nil {
							return nil, err
						}
						bg, err := bgStream.next()
						if err != nil {
							return nil, err
						}
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
			bitType, err := typeBR.readBit()
			if err != nil {
				return nil, err
			}
			fg, err := fgStream.next()
			if err != nil {
				return nil, err
			}
			bg, err := bgStream.next()
			if err != nil {
				return nil, err
			}
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
			bitType, err := typeBR.readBit()
			if err != nil {
				return nil, err
			}
			fg, err := fgStream.next()
			if err != nil {
				return nil, err
			}
			bg, err := bgStream.next()
			if err != nil {
				return nil, err
			}
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

	if blockIndex != int(blockCount) {
		return nil, fmt.Errorf("block count mismatch: used %d of %d", blockIndex, blockCount)
	}
	if fgStream != nil && fgStream.n != int(blockCount) && fgStream.i != int(blockCount) {
		return nil, fmt.Errorf("color stream mismatch: used fg=%d blocks=%d", fgStream.i, blockCount)
	}
	if bgStream != nil && bgStream.n != int(blockCount) && bgStream.i != int(blockCount) {
		return nil, fmt.Errorf("color stream mismatch: used bg=%d blocks=%d", bgStream.i, blockCount)
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

	// read channel segments sequentially from the payload slice.
	// Y is always present; Cb/Cr may be omitted in grayscale mode.
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

	// Y is always decoded.
	wg.Add(1)
	go func() {
		defer wg.Done()
		resY.plane, resY.err = decodeChannel(ySeg, imgW, imgH)
	}()

	if hasCb {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resCb.plane, resCb.err = decodeChannel(cbSeg, imgW, imgH)
		}()
	}
	if hasCr {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resCr.plane, resCr.err = decodeChannel(crSeg, imgW, imgH)
		}()
	}

	wg.Wait()

	if resY.err != nil {
		return nil, resY.err
	}

	yPlane := resY.plane

	// If Cb/Cr are missing, synthesise neutral chroma=128 so that the
	// resulting RGB image is grayscale.
	var cbPlane, crPlane []uint8
	if hasCb {
		if resCb.err != nil {
			return nil, resCb.err
		}
		cbPlane = resCb.plane
	} else {
		cbPlane = make([]uint8, imgW*imgH)
		for i := range cbPlane {
			cbPlane[i] = 128
		}
	}

	if hasCr {
		if resCr.err != nil {
			return nil, resCr.err
		}
		crPlane = resCr.plane
	} else {
		crPlane = make([]uint8, imgW*imgH)
		for i := range crPlane {
			crPlane[i] = 128
		}
	}

	// combine into RGBA (fast path: write directly into Pix) using integer-based YCbCr->RGB conversion
	dst := image.NewRGBA(image.Rect(0, 0, imgW, imgH))
	pix := dst.Pix
	stride := dst.Stride

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
		go func(yStart, yEnd int) {
			defer wgRGB.Done()
			for y := yStart; y < yEnd; y++ {
				rowOff := y * stride
				baseIdx := y * imgW
				for x := range imgW {
					idx := baseIdx + x
					Y := int32(yPlane[idx])
					Cb := int32(cbPlane[idx]) - 128
					Cr := int32(crPlane[idx]) - 128

					// Fixed-point BT.601:
					// R ≈ Y + 1.402 * Cr
					// G ≈ Y - 0.344136 * Cb - 0.714136 * Cr
					// B ≈ Y + 1.772 * Cb
					// Coefficients are scaled by 1<<16.
					R := Y + ((91881 * Cr) >> 16)
					G := Y - ((22554*Cb + 46802*Cr) >> 16)
					B := Y + ((116130 * Cb) >> 16)

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
		}(y0, y1)
	}
	wgRGB.Wait()

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

var zstdEncPool = sync.Pool{
	New: func() any {
		enc, _ := zstd.NewWriter(nil)
		return enc
	},
}

var zstdDecPool = sync.Pool{
	New: func() any {
		dec, _ := zstd.NewReader(nil)
		return dec
	},
}

func compressZstd(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	var buf bytes.Buffer

	enc := zstdEncPool.Get().(*zstd.Encoder)
	enc.Reset(&buf)

	if _, err := enc.Write(data); err != nil {
		_ = enc.Close()
		zstdEncPool.Put(enc)
		return nil, err
	}

	if err := enc.Close(); err != nil {
		zstdEncPool.Put(enc)
		return nil, err
	}

	zstdEncPool.Put(enc)
	return buf.Bytes(), nil
}

func decompressZstd(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	dec := zstdDecPool.Get().(*zstd.Decoder)
	if err := dec.Reset(bytes.NewReader(data)); err != nil {
		zstdDecPool.Put(dec)
		return nil, err
	}

	var out bytes.Buffer
	if _, err := out.ReadFrom(dec); err != nil {
		zstdDecPool.Put(dec)
		return nil, err
	}

	zstdDecPool.Put(dec)
	return out.Bytes(), nil
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

func newDeltaStream(packed []byte, n int) (*deltaStream, error) {
	if n == 0 {
		return &deltaStream{}, nil
	}
	if len(packed) < n {
		return nil, fmt.Errorf("delta stream truncated: have %d, need %d", len(packed), n)
	}
	return &deltaStream{
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
