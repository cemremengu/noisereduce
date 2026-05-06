package noisereduce

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHannPeriodic(t *testing.T) {
	w := hannPeriodic(8)
	// Periodic Hann of length 8: 0.5 - 0.5*cos(2*pi*k/8)
	want := []float64{
		0.0,
		0.14644660940672621,
		0.5,
		0.8535533905932737,
		1.0,
		0.8535533905932737,
		0.5,
		0.14644660940672626,
	}
	for i, v := range w {
		assert.InDeltaf(t, want[i], v, 1e-12, "hannPeriodic[%d]", i)
	}
}

func TestSTFTRoundTripSinusoid(t *testing.T) {
	const sr = 16000
	const n = 16000
	const nfft = 1024
	const winLen = 1024
	const hop = 256

	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(2*math.Pi*440*float64(i)/float64(sr)) +
			0.3*math.Sin(2*math.Pi*1320*float64(i)/float64(sr))
	}

	z := stft(x, nfft, winLen, hop)
	y := istft(z, nfft, winLen, hop, n)

	require.Len(t, y, n)

	// Outside the boundary region (one window length on each end), the
	// reconstruction should be essentially exact.
	maxErr := 0.0
	for i := winLen; i < n-winLen; i++ {
		e := math.Abs(x[i] - y[i])
		if e > maxErr {
			maxErr = e
		}
	}
	assert.Less(t, maxErr, 1e-9, "interior max error")
}

func TestSTFTRoundTripNoise(t *testing.T) {
	const n = 8192
	const nfft = 512
	const winLen = 512
	const hop = 128

	r := rand.New(rand.NewSource(7))
	x := make([]float64, n)
	for i := range x {
		x[i] = r.NormFloat64()
	}

	z := stft(x, nfft, winLen, hop)
	y := istft(z, nfft, winLen, hop, n)

	require.Len(t, y, n)
	maxErr := 0.0
	for i := winLen; i < n-winLen; i++ {
		e := math.Abs(x[i] - y[i])
		if e > maxErr {
			maxErr = e
		}
	}
	assert.Less(t, maxErr, 1e-9, "noise round-trip max error")
}

func TestSTFTRoundTripShortSignal(t *testing.T) {
	const n = 100
	const nfft = 1024
	const winLen = 1024
	const hop = 256

	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i + 1)
	}

	z := stft(x, nfft, winLen, hop)
	y := istft(z, nfft, winLen, hop, n)

	require.Len(t, y, n)
	for i := range x {
		assert.InDeltaf(t, x[i], y[i], 1e-9, "[%d]", i)
	}
}

func TestSTFTRoundTripPreservesNonHopAlignedTail(t *testing.T) {
	const n = 1000
	const nfft = 1024
	const winLen = 1024
	const hop = 256

	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i + 1)
	}

	z := stft(x, nfft, winLen, hop)
	y := istft(z, nfft, winLen, hop, n)

	require.Len(t, y, n)
	for i := range x {
		assert.InDeltaf(t, x[i], y[i], 1e-9, "[%d]", i)
	}
}

func TestSTFTPaddedFalseDropsLongNonHopAlignedTail(t *testing.T) {
	const n = 4095
	const nfft = 1024
	const winLen = 1024
	const hop = 256

	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i + 1)
	}

	z := stft(x, nfft, winLen, hop)
	require.Equal(t, 16, len(z[0]), "frames should match no-tail-padding behavior")
	y := istft(z, nfft, winLen, hop, n)

	require.Len(t, y, n)
	covered := (len(z[0]) - 1) * hop
	for i := winLen; i < covered-winLen; i++ {
		assert.InDeltaf(t, x[i], y[i], 1e-9, "[%d]", i)
	}
	for i := covered; i < len(y); i++ {
		assert.Equalf(t, 0.0, y[i], "tail [%d] should be 0 after padded=False drops incomplete tail", i)
	}
}

func TestSTFTBins(t *testing.T) {
	x := make([]float64, 4096)
	for i := range x {
		x[i] = math.Sin(2 * math.Pi * float64(i) / 64)
	}
	z := stft(x, 1024, 1024, 256)
	require.Len(t, z, 1024/2+1)
	require.NotEmpty(t, z[0], "no frames emitted")
}
