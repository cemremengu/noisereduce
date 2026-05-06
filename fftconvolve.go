package noisereduce

import (
	"github.com/madelynnblue/go-dsp/fft"
)

// fftconvolve2DSame computes the 2-D linear convolution of `mask` and
// `kernel` and returns the result cropped to mask's shape. Both inputs
// are rectangular [][]float64 in [rows][cols] order.
//
// Used to smooth the spectrogram mask (mask shape [freqBins][frames],
// kernel shape [(2*n_grad_freq+1)][(2*n_grad_time+1)]).
func fftconvolve2DSame(mask, kernel [][]float64) [][]float64 {
	if len(mask) == 0 || len(kernel) == 0 {
		return mask
	}
	f := len(mask)
	tt := len(mask[0])
	kf := len(kernel)
	kt := len(kernel[0])
	pf := f + kf - 1
	pt := tt + kt - 1

	mPad := zeroPad2D(mask, pf, pt)
	kPad := zeroPad2D(kernel, pf, pt)

	mFFT := fft.FFT2Real(mPad)
	kFFT := fft.FFT2Real(kPad)

	for i := 0; i < pf; i++ {
		for j := 0; j < pt; j++ {
			mFFT[i][j] *= kFFT[i][j]
		}
	}

	yFull := fft.IFFT2(mFFT)

	startF := (kf - 1) / 2
	startT := (kt - 1) / 2

	out := make([][]float64, f)
	for i := 0; i < f; i++ {
		row := make([]float64, tt)
		src := yFull[startF+i]
		for j := 0; j < tt; j++ {
			row[j] = real(src[startT+j])
		}
		out[i] = row
	}
	return out
}

func zeroPad2D(m [][]float64, rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
	}
	for i, row := range m {
		copy(out[i], row)
	}
	return out
}
