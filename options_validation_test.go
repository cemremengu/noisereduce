package noisereduce

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestReduceNoiseRejectsInvalidFFTOptions(t *testing.T) {
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	tests := []struct {
		name string
		opt  Options
	}{
		{
			name: "negative NFFT",
			opt: Options{
				NFFT:      -8,
				WinLength: 4,
				HopLength: 1,
			},
		},
		{
			name: "negative WinLength",
			opt: Options{
				NFFT:      8,
				WinLength: -1,
				HopLength: 1,
			},
		},
		{
			name: "negative HopLength",
			opt: Options{
				NFFT:      8,
				WinLength: 4,
				HopLength: -1,
			},
		},
		{
			name: "HopLength larger than WinLength",
			opt: Options{
				NFFT:      8,
				WinLength: 4,
				HopLength: 5,
			},
		},
		{
			name: "negative FreqMaskSmoothHz",
			opt: Options{
				NFFT:             8,
				WinLength:        4,
				HopLength:        1,
				FreqMaskSmoothHz: -1,
			},
		},
		{
			name: "negative TimeMaskSmoothMs",
			opt: Options{
				NFFT:             8,
				WinLength:        4,
				HopLength:        1,
				TimeMaskSmoothMs: -1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ReduceNoiseMono(y, 16000, tt.opt)
			require.Error(t, err)
		})
	}
}

func TestReduceNoiseRejectsMismatchedYNoiseChannels(t *testing.T) {
	opt := DefaultOptions()
	opt.Algorithm = Stationary
	opt.YNoise = [][]float64{
		{1, 2},
		{1, 2, 3},
	}

	_, err := ReduceNoiseMono([]float64{1, 2, 3, 4}, 16000, opt)
	require.Error(t, err)
}

func TestReduceNoiseRejectsNegativePadding(t *testing.T) {
	opt := DefaultOptions()
	opt.Padding = -1

	_, err := ReduceNoiseMono([]float64{1, 2, 3, 4}, 16000, opt)
	require.Error(t, err)
}
