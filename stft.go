package noisereduce

import (
	"math"

	"github.com/madelynnblue/go-dsp/fft"
)

// hannPeriodic returns the periodic Hann window of length n.
//
//	w[k] = 0.5 - 0.5 * cos(2*pi*k/n), k = 0..n-1
//
// The periodic form preserves COLA at hop = win/4; the symmetric variant
// degrades reconstruction.
func hannPeriodic(n int) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	for k := 0; k < n; k++ {
		w[k] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(k)/float64(n))
	}
	return w
}

// stft computes the one-sided short-time Fourier transform of x:
//
//   - x is left/right zero-padded by nperseg/2 (boundary='zeros').
//   - window is multiplied per frame, segment then zero-padded to nfft.
//   - returned matrix has shape [nfft/2+1][nframes]complex128.
//   - amplitudes are divided by sum(window).
//   - for inputs at least one window long, no analysis-only tail padding is
//     appended, so non-hop-aligned tails are dropped.
//
// nperseg = winLen, noverlap = winLen - hop, so frame stride is `hop`.
// Very short inputs keep ceiling coverage so round trips remain robust.
func stft(x []float64, nfft, winLen, hop int) [][]complex128 {
	if winLen > nfft {
		panic("noisereduce: winLen > nfft")
	}
	w := hannPeriodic(winLen)
	winSum := 0.0
	for _, v := range w {
		winSum += v
	}

	pad := winLen / 2
	padded := make([]float64, len(x)+2*pad)
	copy(padded[pad:], x)

	bodyLen := len(x)
	minOutLen := bodyLen + 2*pad
	nframes := 1
	if bodyLen >= winLen {
		// Emit centers 0, hop, 2*hop, ... through
		// floor(len(x)/hop)*hop. Any incomplete tail after that point is
		// not analysed.
		nframes += bodyLen / hop
	} else if minOutLen > winLen {
		// Preserve robust short-input round trips.
		nframes += (minOutLen - winLen + hop - 1) / hop
	}
	neededLen := (nframes-1)*hop + winLen
	if len(padded) < neededLen {
		extra := neededLen - len(padded)
		padded = append(padded, make([]float64, extra)...)
	}

	bins := nfft/2 + 1
	out := make([][]complex128, bins)
	for i := range out {
		out[i] = make([]complex128, nframes)
	}

	frame := make([]float64, nfft)
	for f := 0; f < nframes; f++ {
		start := f * hop
		// window the segment, zero-pad to nfft
		for k := 0; k < winLen; k++ {
			frame[k] = padded[start+k] * w[k]
		}
		for k := winLen; k < nfft; k++ {
			frame[k] = 0
		}
		spec := fft.FFTReal(frame) // length nfft, conjugate-symmetric
		for b := 0; b < bins; b++ {
			out[b][f] = complex(real(spec[b])/winSum, imag(spec[b])/winSum)
		}
	}
	return out
}

// istft is the inverse of stft. The trim length matches the original
// input length when winLen, hop, and the input length are consistent
// with the same parameters used in stft (i.e. signalLen samples were
// passed to stft).
func istft(z [][]complex128, nfft, winLen, hop int, signalLen int) []float64 {
	if winLen > nfft {
		panic("noisereduce: winLen > nfft")
	}
	w := hannPeriodic(winLen)
	winSum := 0.0
	for _, v := range w {
		winSum += v
	}

	bins := len(z)
	if bins != nfft/2+1 {
		panic("noisereduce: spectrogram bin count mismatch")
	}
	nframes := 0
	if bins > 0 {
		nframes = len(z[0])
	}

	outLen := (nframes-1)*hop + winLen
	if outLen < 0 {
		outLen = 0
	}
	yPadded := make([]float64, outLen)
	wsq := make([]float64, outLen)

	// Pre-compute window^2 for COLA normalisation.
	w2 := make([]float64, winLen)
	for k := range w {
		w2[k] = w[k] * w[k]
	}

	// Per-frame inverse FFT. fft.IFFTReal expects a real-valued slice;
	// we have a half-spectrum, so fall back to fft.IFFT on the full
	// reconstructed conjugate-symmetric vector.
	full := make([]complex128, nfft)
	for f := 0; f < nframes; f++ {
		// rebuild full conjugate-symmetric spectrum, scale back by winSum
		for b := 0; b < bins; b++ {
			c := z[b][f]
			full[b] = complex(real(c)*winSum, imag(c)*winSum)
		}
		for b := bins; b < nfft; b++ {
			c := full[nfft-b]
			full[b] = complex(real(c), -imag(c))
		}
		td := fft.IFFT(full)

		start := f * hop
		for k := 0; k < winLen; k++ {
			yPadded[start+k] += real(td[k]) * w[k]
			wsq[start+k] += w2[k]
		}
	}

	// Normalise by accumulated window-squared (COLA denominator).
	const eps = 1e-12
	for i := range yPadded {
		if wsq[i] > eps {
			yPadded[i] /= wsq[i]
		}
	}

	// Trim the boundary='zeros' pad applied in stft (winLen/2 each side).
	pad := winLen / 2
	if pad > len(yPadded) {
		return nil
	}
	body := yPadded[pad : len(yPadded)-pad]
	if signalLen > 0 {
		if signalLen <= len(body) {
			return body[:signalLen]
		}
		out := make([]float64, signalLen)
		copy(out, body)
		return out
	}
	return body
}
