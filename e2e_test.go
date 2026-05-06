package noisereduce

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const fixtureWAV = "assets/fish.wav"

func loadFixture(t *testing.T) ([]float64, int) {
	t.Helper()
	path, err := filepath.Abs(fixtureWAV)
	require.NoError(t, err, "resolve fixture path")
	channels, sr, err := ReadWAV(path)
	require.NoError(t, err, "read fixture %s", path)
	require.NotEmpty(t, channels, "fixture has no audio")
	return channels[0], sr
}

func rms(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	s := 0.0
	for _, v := range x {
		s += v * v
	}
	return math.Sqrt(s / float64(len(x)))
}

func addNoisySignal(t *testing.T, seed int64) (clean, noisy, scaledNoise []float64, sr int) {
	t.Helper()
	clean, sr = loadFixture(t)
	r := rand.New(rand.NewSource(seed))
	noise := BandLimitedNoise(2000, 12000, len(clean), sr, r)
	// Match the reference test multiplier while scaling to this fixture's float range.
	scale := 10.0 * rms(clean) / (rms(noise) + 1e-12)
	scaledNoise = make([]float64, len(noise))
	noisy = make([]float64, len(clean))
	for i := range noisy {
		scaledNoise[i] = noise[i] * scale
		noisy[i] = clean[i] + scaledNoise[i]
	}
	return clean, noisy, scaledNoise, sr
}

func TestReduceNoiseStationary(t *testing.T) {
	_, noisy, _, sr := addNoisySignal(t, 42)
	opt := DefaultOptions()
	opt.Algorithm = Stationary
	opt.NJobs = 2
	out, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err, "ReduceNoise")
	require.Len(t, out, len(noisy))
	require.True(t, allFinite(out), "output has non-finite values")
	assert.Less(t, rms(out), rms(noisy), "expected RMS to drop")
}

func TestReduceNoiseStationaryWithNoiseClip(t *testing.T) {
	_, noisy, scaledNoise, sr := addNoisySignal(t, 42)
	const noiseLenSec = 2
	clipLen := sr * noiseLenSec
	if clipLen > len(scaledNoise) {
		clipLen = len(scaledNoise)
	}
	noiseClip := scaledNoise[:clipLen]

	opt := DefaultOptions()
	opt.Algorithm = Stationary
	opt.YNoise = [][]float64{noiseClip}
	opt.NJobs = 2
	out, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err, "ReduceNoise")
	require.Len(t, out, len(noisy))
	require.True(t, allFinite(out), "output has non-finite values")
	assert.Less(t, rms(out), rms(noisy), "expected RMS to drop")

	noClip := opt
	noClip.YNoise = nil
	other, err := ReduceNoiseMono(noisy, sr, noClip)
	require.NoError(t, err)
	differs := false
	for i := range out {
		if math.Abs(out[i]-other[i]) > 1e-9 {
			differs = true
			break
		}
	}
	assert.True(t, differs, "YNoise had no effect on output")
}

func TestReduceNoiseNonStationary(t *testing.T) {
	_, noisy, _, sr := addNoisySignal(t, 42)
	opt := DefaultOptions()
	opt.Algorithm = NonStationary
	opt.NJobs = 2
	out, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err, "ReduceNoise")
	require.Len(t, out, len(noisy))
	require.True(t, allFinite(out), "output has non-finite values")
	assert.Less(t, rms(out), rms(noisy), "expected RMS to drop")
}

func TestReduceNoiseNonStationaryRejectsNegativeTimeConstant(t *testing.T) {
	opt := DefaultOptions()
	opt.Algorithm = NonStationary
	opt.TimeConstantS = -1

	_, err := ReduceNoiseMono([]float64{0, 1, 0, -1}, 16000, opt)
	require.Error(t, err, "ReduceNoiseMono accepted negative TimeConstantS")
}

func TestReduceNoiseDeterministic(t *testing.T) {
	_, noisy, _, sr := addNoisySignal(t, 42)
	opt := DefaultOptions()
	opt.NJobs = 1
	a, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err)
	b, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err)
	for i := range a {
		assert.Equalf(t, a[i], b[i], "non-deterministic at [%d]", i)
	}
}

func TestReduceNoiseChunked(t *testing.T) {
	_, noisy, _, sr := addNoisySignal(t, 42)
	opt := DefaultOptions()
	opt.NJobs = 2
	opt.ChunkSize = 30000
	opt.Padding = 10000
	out, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err)
	require.Len(t, out, len(noisy))
	require.True(t, allFinite(out), "non-finite output")
	assert.Less(t, rms(out), rms(noisy), "expected RMS to drop")
}

func TestPropDecreaseZeroIsIdentity(t *testing.T) {
	_, noisy, _, sr := addNoisySignal(t, 42)
	opt := DefaultOptions()
	opt.PropDecrease = 0
	opt.NJobs = 1
	opt.ChunkSize = 0
	out, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err)
	require.Len(t, out, len(noisy))

	winLen := opt.WinLength
	if winLen == 0 {
		winLen = opt.NFFT
	}
	hop := opt.HopLength
	if hop == 0 {
		hop = winLen / 4
	}
	z := stft(noisy, opt.NFFT, winLen, hop)
	ref := istft(z, opt.NFFT, winLen, hop, len(noisy))
	maxErr := 0.0
	for i := winLen; i < len(out)-winLen; i++ {
		if e := math.Abs(out[i] - ref[i]); e > maxErr {
			maxErr = e
		}
	}
	assert.Less(t, maxErr, 1e-9, "PropDecrease=0 should be a near-identity blend")
}

func TestPropDecreaseZeroPaddingZeroPreservesShortInput(t *testing.T) {
	noisy := make([]float64, 100)
	for i := range noisy {
		noisy[i] = float64(i + 1)
	}
	opt := DefaultOptions()
	opt.PropDecrease = 0
	opt.Padding = 0
	opt.ChunkSize = 0
	opt.NJobs = 1

	out, err := ReduceNoiseMono(noisy, 16000, opt)
	require.NoError(t, err)
	require.Len(t, out, len(noisy))
	for i := range noisy {
		assert.InDeltaf(t, noisy[i], out[i], 1e-9, "[%d]", i)
	}
}

func TestStationaryPropDecreaseZeroWithoutSmoothingIsIdentity(t *testing.T) {
	const sr = 16000
	y := make([]float64, 4096)
	for i := range y {
		y[i] = 1 + 0.25*math.Sin(2*math.Pi*55*float64(i)/sr)
	}

	opt := DefaultOptions()
	opt.Algorithm = Stationary
	opt.PropDecrease = 0
	opt.Padding = 0
	opt.ChunkSize = 0
	opt.NJobs = 1
	opt.FreqMaskSmoothHz = 0
	opt.TimeMaskSmoothMs = 0

	out, err := ReduceNoiseMono(y, sr, opt)
	require.NoError(t, err)
	require.Len(t, out, len(y))

	winLen := opt.WinLength
	if winLen == 0 {
		winLen = opt.NFFT
	}
	hop := opt.HopLength
	if hop == 0 {
		hop = winLen / 4
	}
	ref := istft(stft(y, opt.NFFT, winLen, hop), opt.NFFT, winLen, hop, len(y))
	maxErr := 0.0
	for i := winLen; i < len(out)-winLen; i++ {
		if e := math.Abs(out[i] - ref[i]); e > maxErr {
			maxErr = e
		}
	}
	assert.Less(t, maxErr, 1e-9, "interior max error")
}

func TestStationaryPropDecreaseZeroWithSmoothingUsesReferenceOrder(t *testing.T) {
	const sr = 16000
	y := make([]float64, 4096)
	for i := range y {
		y[i] = 1 + 0.25*math.Sin(2*math.Pi*55*float64(i)/sr)
	}

	opt := DefaultOptions()
	opt.Algorithm = Stationary
	opt.PropDecrease = 0
	opt.Padding = 0
	opt.ChunkSize = 0
	opt.NJobs = 1

	out, err := ReduceNoiseMono(y, sr, opt)
	require.NoError(t, err)
	require.Len(t, out, len(y))
	require.True(t, allFinite(out), "non-finite output")

	winLen := opt.WinLength
	if winLen == 0 {
		winLen = opt.NFFT
	}
	hop := opt.HopLength
	if hop == 0 {
		hop = winLen / 4
	}
	ref := istft(stft(y, opt.NFFT, winLen, hop), opt.NFFT, winLen, hop, len(y))
	maxErr := 0.0
	for i := range out {
		if e := math.Abs(out[i] - ref[i]); e > maxErr {
			maxErr = e
		}
	}
	assert.Greater(t, maxErr, 1e-3, "want noticeable edge smoothing; blend before smoothing")
}

func TestChunkSizeZeroDisablesChunking(t *testing.T) {
	_, noisy, _, sr := addNoisySignal(t, 42)
	const minLen = 650_000 // > old default ChunkSize of 600_000
	if len(noisy) < minLen {
		tiled := make([]float64, 0, minLen+len(noisy))
		for len(tiled) < minLen {
			tiled = append(tiled, noisy...)
		}
		noisy = tiled[:minLen]
	}
	opt := DefaultOptions()
	opt.NJobs = 1
	opt.Algorithm = Stationary

	opt.ChunkSize = 0
	unchunked, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err)
	opt.ChunkSize = len(noisy) + 1
	oneChunk, err := ReduceNoiseMono(noisy, sr, opt)
	require.NoError(t, err)
	require.Len(t, oneChunk, len(unchunked))
	for i := range unchunked {
		assert.InDeltaf(t, oneChunk[i], unchunked[i], 1e-12, "ChunkSize=0 diverged from one-shot at [%d]", i)
	}
}

func allFinite(x []float64) bool {
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}
