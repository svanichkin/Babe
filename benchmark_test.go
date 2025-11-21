package main

import (
	"bytes"
	"image"
	"image/jpeg"
	"os"
	"testing"
)

func loadTestImage(b *testing.B) image.Image {
	f, err := os.Open("test.jpg")
	if err != nil {
		b.Fatalf("failed to open test.jpg: %v", err)
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		b.Fatalf("failed to decode test.jpg: %v", err)
	}
	return img
}

func BenchmarkJPEG(b *testing.B) {
	img := loadTestImage(b)

	buf := &bytes.Buffer{}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		buf.Reset()
		if err := jpeg.Encode(buf, img, &jpeg.Options{Quality: 80}); err != nil {
			b.Fatalf("jpeg encode failed: %v", err)
		}
	}
}

func BenchmarkBABE(b *testing.B) {
	img := loadTestImage(b)

	// bounds := img.Bounds()
	// cols := bounds.Dx() / 4
	// rows := bounds.Dy() / 8

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, err := Encode(img, 60); err != nil {
			b.Fatalf("tiv encode failed: %v", err)
		}
	}
}
