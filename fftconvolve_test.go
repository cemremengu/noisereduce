package noisereduce

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// directConv2DSame is a brute-force reference implementation used to
// verify fftconvolve2DSame.
func directConv2DSame(mask, kernel [][]float64) [][]float64 {
	f := len(mask)
	tt := len(mask[0])
	kf := len(kernel)
	kt := len(kernel[0])
	pf := f + kf - 1
	pt := tt + kt - 1
	full := make([][]float64, pf)
	for i := range full {
		full[i] = make([]float64, pt)
	}
	for i := 0; i < f; i++ {
		for j := 0; j < tt; j++ {
			v := mask[i][j]
			if v == 0 {
				continue
			}
			for ki := 0; ki < kf; ki++ {
				for kj := 0; kj < kt; kj++ {
					full[i+ki][j+kj] += v * kernel[ki][kj]
				}
			}
		}
	}
	startF := (kf - 1) / 2
	startT := (kt - 1) / 2
	out := make([][]float64, f)
	for i := 0; i < f; i++ {
		row := make([]float64, tt)
		copy(row, full[startF+i][startT:startT+tt])
		out[i] = row
	}
	return out
}

func TestFFTConvolve2DSameMatchesDirect(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	mask := make([][]float64, 5)
	for i := range mask {
		mask[i] = make([]float64, 7)
		for j := range mask[i] {
			mask[i][j] = r.Float64()
		}
	}
	kernel := make([][]float64, 3)
	for i := range kernel {
		kernel[i] = make([]float64, 3)
		for j := range kernel[i] {
			kernel[i][j] = r.Float64()
		}
	}
	got := fftconvolve2DSame(mask, kernel)
	want := directConv2DSame(mask, kernel)
	require.Len(t, got, len(want), "row count mismatch")
	require.Len(t, got[0], len(want[0]), "column count mismatch")
	for i := range want {
		for j := range want[i] {
			assert.InDeltaf(t, want[i][j], got[i][j], 1e-9, "[%d,%d]", i, j)
		}
	}
}

func TestFFTConvolve2DSameDeltaKernel(t *testing.T) {
	mask := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}
	// 3x3 identity (delta) kernel — output should equal the input.
	kernel := [][]float64{
		{0, 0, 0},
		{0, 1, 0},
		{0, 0, 0},
	}
	got := fftconvolve2DSame(mask, kernel)
	for i := range mask {
		for j := range mask[i] {
			assert.InDeltaf(t, mask[i][j], got[i][j], 1e-9, "[%d,%d]", i, j)
		}
	}
}
