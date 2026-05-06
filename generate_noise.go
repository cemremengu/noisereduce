package noisereduce

import (
	"math"
	"math/rand"

	"github.com/madelynnblue/go-dsp/fft"
)

// FFTNoise randomises phases in a real-valued half-symmetric spectrum f
// and returns the inverse FFT real part.
//
// f is a length-N real spectrum (typically a band-pass mask). The
// output is a length-N real time-domain signal.
func FFTNoise(f []float64, r *rand.Rand) []float64 {
	if r == nil {
		r = rand.New(rand.NewSource(0)) // #nosec G404 -- deterministic DSP noise, not security-sensitive randomness.
	}
	n := len(f)
	c := make([]complex128, n)
	for i, v := range f {
		c[i] = complex(v, 0)
	}
	np := (n - 1) / 2
	for i := 1; i <= np; i++ {
		theta := r.Float64() * 2 * math.Pi
		phase := complex(math.Cos(theta), math.Sin(theta))
		c[i] *= phase
		// Conjugate-symmetric counterpart at index n-i.
		c[n-i] = complex(real(c[i]), -imag(c[i]))
	}
	td := fft.IFFT(c)
	out := make([]float64, n)
	for i, v := range td {
		out[i] = real(v)
	}
	return out
}

// BandLimitedNoise generates a band-limited noise signal of the given
// length and sample rate.
//
// minFreq and maxFreq are inclusive Hz bounds.
func BandLimitedNoise(minFreq, maxFreq float64, samples, sampleRate int, r *rand.Rand) []float64 {
	freqs := fftFreqAbs(samples, 1.0/float64(sampleRate))
	f := make([]float64, samples)
	for i, fr := range freqs {
		if fr >= minFreq && fr <= maxFreq {
			f[i] = 1
		}
	}
	return FFTNoise(f, r)
}

// fftFreqAbs returns the absolute FFT bin frequencies:
//
//	[0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d)        (n even)
//	[0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (n*d)  (n odd)
//
// We just need the magnitudes, since the band mask is symmetric.
func fftFreqAbs(n int, d float64) []float64 {
	out := make([]float64, n)
	denom := float64(n) * d
	half := n / 2
	for i := 0; i < n; i++ {
		var k int
		if n%2 == 0 {
			if i < half {
				k = i
			} else {
				k = i - n
			}
		} else {
			if i <= (n-1)/2 {
				k = i
			} else {
				k = i - n
			}
		}
		f := float64(k) / denom
		if f < 0 {
			f = -f
		}
		out[i] = f
	}
	return out
}
