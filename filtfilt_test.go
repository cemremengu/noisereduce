package noisereduce

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFiltfilt1PoleConstantInput(t *testing.T) {
	x := make([]float64, 64)
	for i := range x {
		x[i] = 3.14
	}
	out := filtfilt1Pole(0.1, x)
	for i, v := range out {
		assert.InDeltaf(t, 3.14, v, 1e-12, "out[%d]", i)
	}
}

func TestFiltfilt1PoleZeroInput(t *testing.T) {
	x := make([]float64, 32)
	out := filtfilt1Pole(0.05, x)
	for i, v := range out {
		assert.Equalf(t, 0.0, v, "out[%d]", i)
	}
}

func TestFiltfilt1PoleManualReference(t *testing.T) {
	// Hand-computed reference for b=0.5, x=[1,2,3,4,5]:
	// Forward:  [1.0, 1.5, 2.25, 3.125, 4.0625]
	// Backward: [1.60546875, 2.2109375, 2.921875, 3.59375, 4.0625]
	x := []float64{1, 2, 3, 4, 5}
	want := []float64{1.60546875, 2.2109375, 2.921875, 3.59375, 4.0625}
	got := filtfilt1Pole(0.5, x)
	for i := range want {
		assert.InDeltaf(t, want[i], got[i], 1e-12, "got[%d]", i)
	}
}

func TestFiltfilt1PoleAttenuatesAlternating(t *testing.T) {
	// 2000 samples >> 5 * time-constant (100), so transients fully decay.
	x := make([]float64, 2000)
	for i := range x {
		if i%2 == 0 {
			x[i] = 1
		} else {
			x[i] = -1
		}
	}
	out := filtfilt1Pole(0.01, x) // very heavy smoothing
	// In the interior, well past the transient, output should be tiny.
	maxAbs := 0.0
	for i := 800; i < len(out)-800; i++ {
		if a := math.Abs(out[i]); a > maxAbs {
			maxAbs = a
		}
	}
	assert.Less(t, maxAbs, 0.05, "interior max should be < 0.05")
}

func TestTimeSmoothCoefficient(t *testing.T) {
	// time_constant=2.0, sr=16000, hop=256:
	//   t = 125; b = (sqrt(1+4*t^2)-1)/(2*t^2)
	b := timeSmoothCoefficient(2.0, 16000, 256)
	t2 := 125.0 * 125.0
	want := (math.Sqrt(1+4*t2) - 1) / (2 * t2)
	assert.InDelta(t, want, b, 1e-15)
	// Edge case: tiny time constant -> pass-through.
	assert.Equal(t, 1.0, timeSmoothCoefficient(0, 16000, 256))
}
