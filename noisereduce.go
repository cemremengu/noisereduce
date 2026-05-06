package noisereduce

import (
	"errors"
	"fmt"
)

// Algorithm selects between stationary and non-stationary spectral gating.
type Algorithm int

const (
	// NonStationary estimates the noise floor continuously from a
	// time-smoothed magnitude spectrogram. Default.
	NonStationary Algorithm = iota
	// Stationary uses a fixed threshold derived from a noise clip
	// (or the signal itself if no clip is supplied).
	Stationary
)

// Options controls ReduceNoise.
//
// Start from DefaultOptions() to get the standard spectral-gating
// settings, then override the knobs you care about. A
// zero-valued Options{} is also valid: every field's zero value has a
// well-defined meaning (e.g. PropDecrease=0 disables suppression,
// ChunkSize=0 disables chunking, FreqMaskSmoothHz=0 disables freq
// smoothing). Only NFFT is auto-defaulted, since 0 makes the STFT
// undefined; pass DefaultOptions().NFFT (1024) explicitly if you
// prefer.
//
// Field names use Go exported identifiers for the algorithm parameters.
type Options struct {
	Algorithm Algorithm

	// YNoise is an optional noise reference for the stationary gate.
	// Shape [channels][frames]. Channels are averaged before STFT.
	// If nil, Y itself is used as the noise reference.
	YNoise [][]float64

	PropDecrease              float64 // proportion to suppress (0..1); default 1.0
	TimeConstantS             float64 // IIR time constant in seconds; default 2.0
	FreqMaskSmoothHz          float64 // freq smoothing width (Hz); 0 disables; default 500
	TimeMaskSmoothMs          float64 // time smoothing width (ms); 0 disables; default 50
	ThreshNMultNonstationary  float64 // sigmoid shift; default 2
	SigmoidSlopeNonstationary float64 // sigmoid slope; default 10
	NStdThreshStationary      float64 // stddev multiplier; default 1.5

	ChunkSize int // frames per chunk; 0 disables chunking; default 600000
	Padding   int // pad applied around each chunk; default 30000

	NFFT      int // FFT length; default 1024
	WinLength int // window length; 0 => NFFT
	HopLength int // hop between frames; 0 => WinLength/4

	ClipNoiseStationary bool // stationary: clip noise to ChunkSize. Default true.

	NJobs int // parallel chunk workers; 0 => runtime.GOMAXPROCS(0)
}

// DefaultOptions returns the recommended option set for spectral gating.
func DefaultOptions() Options {
	return Options{
		Algorithm:                 NonStationary,
		PropDecrease:              1.0,
		TimeConstantS:             2.0,
		FreqMaskSmoothHz:          500,
		TimeMaskSmoothMs:          50,
		ThreshNMultNonstationary:  2,
		SigmoidSlopeNonstationary: 10,
		NStdThreshStationary:      1.5,
		ChunkSize:                 600000,
		Padding:                   30000,
		NFFT:                      1024,
		WinLength:                 0,
		HopLength:                 0,
		ClipNoiseStationary:       true,
		NJobs:                     0,
	}
}

// applyDefaults fills in only fields that have no usable zero-value
// semantics, so an explicit override (including 0) survives:
//
//   - NFFT: zero would make the STFT length 0 and divide-by-zero in
//     hop derivation; default to 1024.
//   - WinLength: documented as "0 => NFFT"; derive.
//   - HopLength: documented as "0 => WinLength/4"; derive.
//
// All other fields keep their literal zero value when set explicitly:
// PropDecrease=0 (identity blend), TimeConstantS=0 (pass-through
// smoothing), ChunkSize=0 (no chunking), etc. NJobs=0 means "use
// GOMAXPROCS" and is resolved inside runChunked.
func (o *Options) applyDefaults() {
	if o.NFFT == 0 {
		o.NFFT = 1024
	}
	if o.WinLength == 0 {
		o.WinLength = o.NFFT
	}
	if o.HopLength == 0 {
		o.HopLength = o.WinLength / 4
	}
}

// ReduceNoise applies spectral-gating noise reduction to a multichannel
// signal shaped [channels][frames]. The output has the same shape.
func ReduceNoise(y [][]float64, sr int, opt Options) ([][]float64, error) {
	if len(y) == 0 {
		return nil, errors.New("noisereduce: empty input")
	}
	frames := len(y[0])
	for i, ch := range y {
		if len(ch) != frames {
			return nil, fmt.Errorf("noisereduce: channel %d has %d frames, expected %d", i, len(ch), frames)
		}
	}
	if sr <= 0 {
		return nil, errors.New("noisereduce: sample rate must be positive")
	}
	opt.applyDefaults()
	if opt.Padding < 0 {
		return nil, errors.New("noisereduce: Padding must be non-negative")
	}

	switch opt.Algorithm {
	case Stationary:
		return runStationary(y, sr, opt)
	case NonStationary:
		if opt.TimeConstantS < 0 {
			return nil, errors.New("noisereduce: TimeConstantS must be non-negative")
		}
		return runNonStationary(y, sr, opt)
	default:
		return nil, fmt.Errorf("noisereduce: unknown algorithm %d", opt.Algorithm)
	}
}

// ReduceNoiseMono is a convenience wrapper for single-channel input.
func ReduceNoiseMono(y []float64, sr int, opt Options) ([]float64, error) {
	out, err := ReduceNoise([][]float64{y}, sr, opt)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
