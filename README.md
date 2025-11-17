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

Specify a custom quality (0–29):

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

## License

MIT
