package main

import (
	"bytes"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/xfmoulet/qoi"
)

func loadTestImage(b *testing.B) image.Image {
	f, err := os.Open("benchmark.png")
	if err != nil {
		b.Fatalf("failed to open benchmark.png: %v", err)
	}
	defer f.Close()

	img, err := png.Decode(f)
	if err != nil {
		b.Fatalf("failed to decode benchmark.png: %v", err)
	}
	return img
}

func BenchmarkJPEG(b *testing.B) {
	img := loadTestImage(b)
	cpus := runtime.NumCPU()
	procs := runtime.GOMAXPROCS(0)
	gor := runtime.NumGoroutine()
	b.Logf("cpus=%d gomaxprocs=%d goroutines=%d", cpus, procs, gor)

	buf := &bytes.Buffer{}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		buf.Reset()

		// Encode
		startEnc := time.Now()
		if err := jpeg.Encode(buf, img, &jpeg.Options{Quality: 80}); err != nil {
			b.Fatalf("jpeg encode failed: %v", err)
		}
		encTime := time.Since(startEnc)
		enc := buf.Bytes()
		encSize := len(enc)

		// Decode
		startDec := time.Now()
		if _, err := jpeg.Decode(bytes.NewReader(enc)); err != nil {
			b.Fatalf("jpeg decode failed: %v", err)
		}
		decTime := time.Since(startDec)

		// Log encode/decode times and size
		b.Logf("encode=%v decode=%v size=%d bytes", encTime, decTime, encSize)

		// Prevent compiler optimization
		if encTime == 0 || decTime == 0 {
			b.Fatalf("unexpected zero time")
		}
	}
}

func BenchmarkBABE(b *testing.B) {
	img := loadTestImage(b)
	cpus := runtime.NumCPU()
	procs := runtime.GOMAXPROCS(0)
	gor := runtime.NumGoroutine()
	b.Logf("cpus=%d gomaxprocs=%d goroutines=%d", cpus, procs, gor)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Encode
		startEnc := time.Now()
		enc, err := Encode(img, 80)
		if err != nil {
			b.Fatalf("encode failed: %v", err)
		}
		encTime := time.Since(startEnc)
		encSize := len(enc)
		_ = encSize

		// Decode
		startDec := time.Now()
		_, err = Decode(enc)
		if err != nil {
			b.Fatalf("decode failed: %v", err)
		}
		decTime := time.Since(startDec)

		// Record encoded size in benchmark output
		b.Logf("encode=%v decode=%v size=%d bytes", encTime, decTime, len(enc))

		// Prevent compiler optimization
		if encTime == 0 || decTime == 0 {
			b.Fatalf("unexpected zero time")
		}
	}
}

func BenchmarkQOI(b *testing.B) {
	img := loadTestImage(b)
	cpus := runtime.NumCPU()
	procs := runtime.GOMAXPROCS(0)
	gor := runtime.NumGoroutine()
	b.Logf("cpus=%d gomaxprocs=%d goroutines=%d", cpus, procs, gor)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Encode
		startEnc := time.Now()
		var buf bytes.Buffer
		if err := qoi.Encode(&buf, img); err != nil {
			b.Fatalf("qoi encode failed: %v", err)
		}
		enc := buf.Bytes()
		encTime := time.Since(startEnc)

		// Decode
		startDec := time.Now()
		_, err := qoi.Decode(bytes.NewReader(enc))
		if err != nil {
			b.Fatalf("qoi decode failed: %v", err)
		}
		decTime := time.Since(startDec)

		b.Logf("encode=%v decode=%v size=%d bytes", encTime, decTime, len(enc))

		if encTime == 0 || decTime == 0 {
			b.Fatalf("unexpected zero time")
		}
	}
}
