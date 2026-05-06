package noisereduce

import "math"

const eps = 2.220446049250313e-16 // machine epsilon for float64

// amplitudeToDB converts a complex magnitude spectrogram to dB and
// applies a per-row top_db clamp.
//
// out[i][j] = max( 20*log10(|x[i][j]|+eps), max_j(20*log10(|x[i][:]|+eps)) - top_db )
func amplitudeToDB(x [][]complex128, topDB float64) [][]float64 {
	out := make([][]float64, len(x))
	for i, row := range x {
		out[i] = make([]float64, len(row))
		rowMax := math.Inf(-1)
		for j, c := range row {
			d := 20 * math.Log10(complexAbs(c)+eps)
			out[i][j] = d
			if d > rowMax {
				rowMax = d
			}
		}
		floor := rowMax - topDB
		for j := range out[i] {
			if out[i][j] < floor {
				out[i][j] = floor
			}
		}
	}
	return out
}

func complexAbs(c complex128) float64 {
	return math.Hypot(real(c), imag(c))
}

// sigmoid computes the soft mask curve used by the non-stationary gate.
//
//	sigmoid(x; shift, mult) = 1 / (1 + exp(-(x + shift) * mult))
func sigmoid(x, shift, mult float64) float64 {
	return 1.0 / (1.0 + math.Exp(-(x+shift)*mult))
}

// magnitude returns |X| for each cell.
func magnitude(x [][]complex128) [][]float64 {
	out := make([][]float64, len(x))
	for i, row := range x {
		out[i] = make([]float64, len(row))
		for j, c := range row {
			out[i][j] = complexAbs(c)
		}
	}
	return out
}

// rowMean returns the mean across columns for each row.
func rowMean(m [][]float64) []float64 {
	out := make([]float64, len(m))
	for i, row := range m {
		if len(row) == 0 {
			continue
		}
		s := 0.0
		for _, v := range row {
			s += v
		}
		out[i] = s / float64(len(row))
	}
	return out
}

// rowStd returns the population standard deviation across columns for
// each row.
func rowStd(m [][]float64) []float64 {
	out := make([]float64, len(m))
	for i, row := range m {
		if len(row) == 0 {
			continue
		}
		mean := 0.0
		for _, v := range row {
			mean += v
		}
		mean /= float64(len(row))
		ss := 0.0
		for _, v := range row {
			d := v - mean
			ss += d * d
		}
		out[i] = math.Sqrt(ss / float64(len(row)))
	}
	return out
}
