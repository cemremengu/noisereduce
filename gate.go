package noisereduce

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
)

// chunkFilter denoises a single padded chunk of one channel and
// returns a slice of the same length. Implementations in
// stationary.go and nonstationary.go.
type chunkFilter func(chunk []float64) []float64

// runChunked applies chunked, padded, parallel processing for one channel.
//
// For each output chunk, we read [start-padding, end+padding) from the
// source (with implicit zero-padding off the ends), pass it to filter,
// and copy back the trim region.
func runChunked(y []float64, chunkSize, padding, nJobs int, filter chunkFilter) []float64 {
	n := len(y)
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	if chunkSize <= 0 || n <= chunkSize {
		// One shot.
		fc := filter(readChunk(y, -padding, n+padding))
		copy(out, fc[padding:padding+n])
		return out
	}

	type job struct {
		ich    int
		pos    int
		length int
	}
	ich2 := (n - 1) / chunkSize
	var jobs []job
	pos := 0
	for ich := 0; ich <= ich2; ich++ {
		end0 := chunkSize
		if ich == ich2 {
			end0 = n - ich*chunkSize
		}
		jobs = append(jobs, job{ich: ich, pos: pos, length: end0})
		pos += end0
	}

	if nJobs <= 0 {
		// 0 (or negative) means "use all available cores", matching the
		// documentation on Options.NJobs.
		nJobs = runtime.GOMAXPROCS(0)
	}
	if nJobs > len(jobs) {
		nJobs = len(jobs)
	}

	jobCh := make(chan job)
	var wg sync.WaitGroup
	worker := func() {
		defer wg.Done()
		for j := range jobCh {
			frameStart := j.ich * chunkSize
			frameEnd := frameStart + j.length
			i1 := frameStart - padding
			i2 := frameEnd + padding
			padded := readChunk(y, i1, i2)
			filtered := filter(padded)
			// trim padding off both ends of the filtered chunk
			copy(out[j.pos:j.pos+j.length], filtered[padding:padding+j.length])
		}
	}
	wg.Add(nJobs)
	for i := 0; i < nJobs; i++ {
		go worker()
	}
	for _, j := range jobs {
		jobCh <- j
	}
	close(jobCh)
	wg.Wait()
	return out
}

// readChunk returns y[i1..i2) zero-padded outside the bounds of y.
func readChunk(y []float64, i1, i2 int) []float64 {
	if i2 < i1 {
		return nil
	}
	chunk := make([]float64, i2-i1)
	n := len(y)
	i1b := i1
	if i1b < 0 {
		i1b = 0
	}
	i2b := i2
	if i2b > n {
		i2b = n
	}
	if i2b > i1b {
		copy(chunk[i1b-i1:], y[i1b:i2b])
	}
	return chunk
}

// linspace returns evenly spaced values over the requested interval.
func linspace(start, stop float64, num int, endpoint bool) []float64 {
	if num <= 0 {
		return nil
	}
	if num == 1 {
		return []float64{start}
	}
	denom := float64(num - 1)
	if !endpoint {
		denom = float64(num)
	}
	step := (stop - start) / denom
	out := make([]float64, num)
	for i := 0; i < num; i++ {
		out[i] = start + step*float64(i)
	}
	return out
}

// smoothingFilter builds an outer product of two triangular ramps,
// normalised to sum 1.
func smoothingFilter(nGradFreq, nGradTime int) [][]float64 {
	vF, vT := smoothingFilterAxes(nGradFreq, nGradTime)
	out := make([][]float64, len(vF))
	for i := range vF {
		out[i] = make([]float64, len(vT))
		for j := range vT {
			out[i][j] = vF[i] * vT[j]
		}
	}
	return out
}

func smoothingFilterAxes(nGradFreq, nGradTime int) ([]float64, []float64) {
	return normalisedKernel(triangleRamp(nGradFreq)), normalisedKernel(triangleRamp(nGradTime))
}

func normalisedKernel(v []float64) []float64 {
	out := append([]float64(nil), v...)
	sum := 0.0
	for _, x := range out {
		sum += x
	}
	if sum == 0 {
		return out
	}
	inv := 1.0 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}

// triangleRamp builds the inner triangular vector used by
// _smoothing_filter:
//
//	concat( linspace(0,1,n+1,endpoint=False),
//	        linspace(1,0,n+2) )[1:-1]
//
// which has length 2n+1, peaking at 1 in the middle.
func triangleRamp(n int) []float64 {
	a := linspace(0, 1, n+1, false) // length n+1, last entry is n/(n+1)
	b := linspace(1, 0, n+2, true)  // length n+2, includes 0 endpoint
	concat := append(append([]float64(nil), a...), b...)
	return concat[1 : len(concat)-1] // drop the leading 0 and trailing 0
}

// gateParams holds the resolved STFT/smoothing geometry for a single
// ReduceNoise call.
type gateParams struct {
	nfft, winLen, hop int
	smoothMask        bool
	smoothKernel      [][]float64 // nil if smoothMask=false
	smoothKernelFreq  []float64
	smoothKernelTime  []float64
}

// resolveGateParams decodes Options into a gateParams, including
// computing the smoothing kernel from FreqMaskSmoothHz / TimeMaskSmoothMs.
func resolveGateParams(opt Options, sr int) (gateParams, error) {
	p := gateParams{nfft: opt.NFFT, winLen: opt.WinLength, hop: opt.HopLength}
	if p.winLen == 0 {
		p.winLen = p.nfft
	}
	if p.hop == 0 {
		p.hop = p.winLen / 4
	}
	if p.nfft <= 0 {
		return p, errors.New("noisereduce: NFFT must be positive")
	}
	if p.winLen <= 0 {
		return p, errors.New("noisereduce: WinLength must be positive")
	}
	if p.winLen > p.nfft {
		return p, fmt.Errorf("noisereduce: WinLength (%d) must be <= NFFT (%d)", p.winLen, p.nfft)
	}
	if p.hop <= 0 {
		return p, errors.New("noisereduce: HopLength must be positive")
	}
	if p.hop > p.winLen {
		return p, fmt.Errorf("noisereduce: HopLength (%d) must be <= WinLength (%d)", p.hop, p.winLen)
	}

	// Smoothing-kernel geometry.
	if opt.FreqMaskSmoothHz < 0 {
		return p, errors.New("noisereduce: FreqMaskSmoothHz must be non-negative")
	}
	if opt.TimeMaskSmoothMs < 0 {
		return p, errors.New("noisereduce: TimeMaskSmoothMs must be non-negative")
	}
	if opt.FreqMaskSmoothHz == 0 && opt.TimeMaskSmoothMs == 0 {
		return p, nil
	}
	nGradFreq := 0
	if opt.FreqMaskSmoothHz > 0 {
		nGradFreq = int(opt.FreqMaskSmoothHz / (float64(sr) / (float64(p.nfft) / 2)))
		if nGradFreq < 1 {
			return p, fmt.Errorf("noisereduce: FreqMaskSmoothHz needs to be at least %.0f Hz",
				float64(sr)/(float64(p.nfft)/2))
		}
	}
	nGradTime := 0
	if opt.TimeMaskSmoothMs > 0 {
		nGradTime = int(opt.TimeMaskSmoothMs / ((float64(p.hop) / float64(sr)) * 1000))
		if nGradTime < 1 {
			return p, fmt.Errorf("noisereduce: TimeMaskSmoothMs needs to be at least %.0f ms",
				(float64(p.hop)/float64(sr))*1000)
		}
	}
	if nGradFreq == 0 && nGradTime == 0 {
		return p, nil
	}
	p.smoothMask = true
	p.smoothKernelFreq, p.smoothKernelTime = smoothingFilterAxes(nGradFreq, nGradTime)
	p.smoothKernel = smoothingFilter(nGradFreq, nGradTime)
	return p, nil
}

func smoothMaskSame(mask [][]float64, gp gateParams) [][]float64 {
	if len(gp.smoothKernelFreq) == 0 || len(gp.smoothKernelTime) == 0 {
		return fftconvolve2DSame(mask, gp.smoothKernel)
	}
	return convolveTimeSame(convolveFreqSame(mask, gp.smoothKernelFreq), gp.smoothKernelTime)
}

func convolveFreqSame(mask [][]float64, kernel []float64) [][]float64 {
	if len(mask) == 0 || len(kernel) == 0 {
		return mask
	}
	f := len(mask)
	tt := len(mask[0])
	start := (len(kernel) - 1) / 2
	out := make([][]float64, f)
	for i := 0; i < f; i++ {
		row := make([]float64, tt)
		for ki, kv := range kernel {
			src := start + i - ki
			if src < 0 || src >= f || kv == 0 {
				continue
			}
			srcRow := mask[src]
			for j, v := range srcRow {
				row[j] += v * kv
			}
		}
		out[i] = row
	}
	return out
}

func convolveTimeSame(mask [][]float64, kernel []float64) [][]float64 {
	if len(mask) == 0 || len(kernel) == 0 {
		return mask
	}
	start := (len(kernel) - 1) / 2
	out := make([][]float64, len(mask))
	for i, srcRow := range mask {
		row := make([]float64, len(srcRow))
		for j := range row {
			sum := 0.0
			for ki, kv := range kernel {
				src := start + j - ki
				if src >= 0 && src < len(srcRow) {
					sum += srcRow[src] * kv
				}
			}
			row[j] = sum
		}
		out[i] = row
	}
	return out
}

// applyMask multiplies each STFT cell by mask[i][j] in place.
func applyMask(z [][]complex128, mask [][]float64) {
	for i := range z {
		for j := range z[i] {
			m := mask[i][j]
			z[i][j] = complex(real(z[i][j])*m, imag(z[i][j])*m)
		}
	}
}

// applyPropDecrease blends the mask between full-on (1) and the
// computed mask: mask <- mask*p + (1-p), in place.
func applyPropDecrease(mask [][]float64, p float64) {
	q := 1.0 - p
	for i := range mask {
		for j := range mask[i] {
			mask[i][j] = mask[i][j]*p + q
		}
	}
}
