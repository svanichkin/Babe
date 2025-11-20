package main

import (
	"fmt"
	"image"
	"image/png"
	_ "image/jpeg"
	_ "image/gif"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func main() {
	if len(os.Args) < 2 || len(os.Args) > 3 {
		fmt.Fprint(os.Stderr, "Encode: babe <input-image> [quality 0–100]\nDecode: babe <input.babe>\n")
		os.Exit(1)
	}

	inputPath := os.Args[1]
	ext := strings.ToLower(filepath.Ext(inputPath))
	base := strings.TrimSuffix(inputPath, filepath.Ext(inputPath))

	// If input is .babe → decode to PNG
	if ext == ".babe" {
		if err := decodeBabe(inputPath, base+".png"); err != nil {
			fmt.Fprintln(os.Stderr, "decode error:", err)
			os.Exit(1)
		}
		fmt.Printf("Decoded %s → %s\n", inputPath, base+".png")
		return
	}

	// Otherwise: encode image → .babe with default or provided quality
	quality := 10
	if len(os.Args) == 3 {
		q, err := strconv.Atoi(os.Args[2])
		if err != nil {
			fmt.Fprintln(os.Stderr, "quality must be an integer between 0 and 100")
			os.Exit(1)
		}
		if q < 0 || q > 100 {
			fmt.Fprintln(os.Stderr, "quality must be between 0 and 100")
			os.Exit(1)
		}
		quality = q
	}

	outPath := base + ".babe"
	if err := encodeToBabe(inputPath, outPath, quality); err != nil {
		fmt.Fprintln(os.Stderr, "encode error:", err)
		os.Exit(1)
	}
	info, err := os.Stat(outPath)
	if err != nil {
		fmt.Printf("Encoded %s (quality=%d) → %s\n", inputPath, quality, outPath)
	} else {
		size := info.Size()
		var sizeStr string
		if size < 1_000_000 {
			sizeStr = fmt.Sprintf("%.2f KB", float64(size)/1024)
		} else {
			sizeStr = fmt.Sprintf("%.2f MB", float64(size)/1024/1024)
		}
		fmt.Printf("Encoded %s (quality=%d, size=%s) → %s\n",
			inputPath, quality, sizeStr, outPath)
	}
}

func encodeToBabe(inPath, outPath string, quality int) error {
	in, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer in.Close()

	img, _, err := image.Decode(in)
	if err != nil {
		return err
	}

	enc, err := Encode(img, quality)
	if err != nil {
		return err
	}

	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := out.Write(enc); err != nil {
		return err
	}

	return nil
}

func decodeBabe(inPath, outPath string) error {
	in, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer in.Close()

	compData, err := io.ReadAll(in)
	if err != nil {
		return err
	}

	dec, err := Decode(compData)
	if err != nil {
		return err
	}

	out, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer out.Close()

	if err := png.Encode(out, dec); err != nil {
		return err
	}

	return nil
}
