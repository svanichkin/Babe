# Babe — Bi-Level Adaptive Block Encoding

**Babe** is an experimental dual‑tone block-based image codec designed for extremely lightweight compression with visually smooth output.  
It focuses on simplicity, compact encoded size, and fast decoding rather than perfect fidelity.

## Features

- Two‑tone block encoding with adaptive subdivision  
- Palette reduction in YUV space  
- Delta-indexed blocks for compact representation  
- Zstandard used for final compression stage  
- Lossy encoder, PNG output on decode  
- Minimal API: `Encode(image, quality)` and `Decode(data)`

## How It Works

At a high level, Babe works as a lossy, block-based codec with a dual-tone model per block and an indexed palette in YUV space.

1. **Color space and input**
   - The source image is converted to a YUV-like space.
   - Luma and chroma are processed with different sensitivity so that most detail is preserved in brightness while color is simplified more aggressively.

2. **Block partitioning**
   - The image is split into rectangular blocks (macroblocks).
   - Blocks can be further subdivided adaptively depending on local contrast and quality settings.
   - Very flat or low-contrast regions tend to keep larger blocks; detailed regions get smaller blocks.

3. **Dual-tone model**
   - For each block Babe tries to approximate all pixels using only **two representative colors** (a “dual-tone”).
   - A simple pattern (bit mask) inside the block tells, for each pixel, which of the two tones is used.
   - This creates a kind of ordered dither / posterization that looks smooth at a distance but is cheap to store.

4. **Palette construction in YUV space**
   - Instead of storing raw RGB values per block, Babe builds a global and/or local palette in YUV space.
   - Colors are quantized and re-used across blocks, so repeated tones only cost index references, not full 24‑bit triples.
   - The palette layout and index width depend on quality settings and image complexity.

5. **Delta indexing and reuse**
   - Indices into the palette are not stored independently; Babe exploits spatial coherence.
   - Neighboring blocks often share or slightly adjust their tones, so indices can often be stored as **small deltas** from a previous index.
   - This reduces the effective bits per block and helps the entropy stage.

6. **Pattern and metadata encoding**
   - For each block, Babe stores:
     - indices of the two palette colors,
     - a compact pattern describing which tone is used per pixel,
     - optional flags/metadata to describe special cases (e.g. flat blocks).
   - Patterns themselves are chosen from a limited family so they can be encoded in only a few bits when repeated.

7. **Bitstream layout**
   - All palette data, block indices, and patterns are serialized into a compact binary stream.
   - The layout is designed for fast sequential decoding: you can reconstruct block tones and patterns with minimal branching.

8. **Final compression (Zstandard)**
   - The raw Babe bitstream is passed through Zstandard.
   - Babe’s structure (reused indices, repeating patterns, small deltas) is optimized to be very friendly to a general‑purpose compressor.
   - This final stage usually cuts the already compact stream by a significant factor.

9. **Decoding**
   - Decode reverses the process:
     - Zstandard decompression to restore the Babe bitstream.
     - Rebuild palettes and per-block parameters.
     - Reconstruct each block’s two tones and pattern.
     - Stitch all blocks into a full‑resolution image in YUV space and convert back to RGB.
   - The operations are mostly table lookups and simple math, which is why decoding is very fast.

This description is simplified; the actual implementation contains additional heuristics and tuning for block sizes, palette limits, and quality parameters to balance speed, size, and visual quality.

## CLI Utility

The repository includes a command-line tool for encoding and decoding.

### Encode an image → `.babe`

```
babe input.jpg
```

Produces:

```
input.babe
```

Default quality is **10**.

Specify a custom quality (0–100):

```
babe input.jpg 5
```

### Decode `.babe` → PNG

```
babe input.babe
```

Produces:

```
input.png
```

## API Usage

### Encode

```go
comp, err := Encode(img, quality)
if err != nil {
    // handle error
}
```

### Decode

```go
img, err := Decode(compData)
if err != nil {
    // handle error
}
```

`Decode` returns a standard `image.Image`.


## Status

This codec is currently experimental.  
Format details, quality tuning, and performance optimizations are still evolving.

## Benchmark Comparison

Approximate performance on Apple M3 (single‑threaded encode/decode):

| Metric | JPEG | BABE | QOI |
|--------|------|------|------|
| **Encode time** | 1.22 s | 0.59 s | 0.86 s |
| **Decode time** | 0.45 s | 0.17–0.20 s | 0.27 s |
| **Total (encode+decode)** | 1.67 s | ~0.77 s | 1.13 s |
| **Output size** | 10.6 MB | 18.2 MB | 60.6 MB |

These numbers are based on internal benchmark tests and may vary depending on hardware and input images.

## License

MIT
