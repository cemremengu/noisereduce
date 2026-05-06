package noisereduce

import (
	"math"
	"math/rand"
	"testing"
)

func benchmarkSignal(n, sr int) []float64 {
	r := rand.New(rand.NewSource(1))
	y := make([]float64, n)
	for i := range y {
		t := float64(i) / float64(sr)
		y[i] = 0.2*math.Sin(2*math.Pi*440*t) + 0.05*(r.Float64()*2-1)
	}
	return y
}

func BenchmarkReduceNoiseNonStationary200k(b *testing.B) {
	y := [][]float64{benchmarkSignal(200_000, 44_100)}
	opt := DefaultOptions()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ReduceNoise(y, 44_100, opt); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReduceNoiseStationary200k(b *testing.B) {
	y := [][]float64{benchmarkSignal(200_000, 44_100)}
	opt := DefaultOptions()
	opt.Algorithm = Stationary
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ReduceNoise(y, 44_100, opt); err != nil {
			b.Fatal(err)
		}
	}
}
