package noisereduce

import "math"

// filtfilt1Pole applies a forward-backward 1-pole IIR filter:
//
//	b_coef = [b], a_coef = [1, b-1], padtype=None
//
// which corresponds to the 1st-order IIR low-pass
//
//	y[n] = b * x[n] + (1 - b) * y[n-1]
//
// applied forward then backward (zero-phase). The effective initial
// condition is y[-1] = x[0] on the forward pass and y[N] =
// y_forward[N-1] on the backward pass.
//
// Result is independent of x's length sign, but length must be >= 1.
func filtfilt1Pole(b float64, x []float64) []float64 {
	n := len(x)
	if n == 0 {
		return nil
	}
	out := make([]float64, n)
	if n == 1 {
		out[0] = x[0]
		return out
	}
	a := 1.0 - b

	// Forward pass: y[-1] = x[0], y[n] = b*x[n] + (1-b)*y[n-1].
	y := make([]float64, n)
	y[0] = b*x[0] + a*x[0] // == x[0]
	for i := 1; i < n; i++ {
		y[i] = b*x[i] + a*y[i-1]
	}

	// Backward pass: u[N-1] = y[N-1], u[n] = b*y[n] + (1-b)*u[n+1].
	out[n-1] = y[n-1]
	for i := n - 2; i >= 0; i-- {
		out[i] = b*y[i] + a*out[i+1]
	}
	return out
}

// timeSmoothCoefficient returns b for the IIR low-pass corresponding to
// `TimeConstantS`.
//
//	t_frames = time_constant_s * sr / hop
//	b = (sqrt(1 + 4*t_frames^2) - 1) / (2*t_frames^2)
func timeSmoothCoefficient(timeConstantS float64, sr, hop int) float64 {
	t := timeConstantS * float64(sr) / float64(hop)
	if t >= 0 && t < 1e-9 {
		return 1.0 // pass-through
	}
	return (math.Sqrt(1+4*t*t) - 1) / (2 * t * t)
}
