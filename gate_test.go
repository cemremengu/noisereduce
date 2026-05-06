package noisereduce

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTriangleRamp(t *testing.T) {
	got := triangleRamp(2)
	want := []float64{1.0 / 3, 2.0 / 3, 1.0, 2.0 / 3, 1.0 / 3}
	require.Len(t, got, len(want))
	for i := range want {
		assert.InDeltaf(t, want[i], got[i], 1e-12, "[%d]", i)
	}
}

func TestSmoothingFilterShapeAndSum(t *testing.T) {
	k := smoothingFilter(2, 3)
	require.Len(t, k, 2*2+1)
	require.Len(t, k[0], 2*3+1)
	sum := 0.0
	for _, row := range k {
		for _, v := range row {
			sum += v
		}
	}
	assert.InDelta(t, 1.0, sum, 1e-12)
	// Center is the largest cell.
	mid := k[2][3]
	for i, row := range k {
		for j, v := range row {
			if i == 2 && j == 3 {
				continue
			}
			assert.LessOrEqualf(t, v, mid, "[%d,%d] exceeds center", i, j)
		}
	}
}

func TestSmoothMaskSameSeparableMatchesDirect2D(t *testing.T) {
	mask := [][]float64{
		{0, 1, 0, 1},
		{1, 0, 1, 0},
		{0, 0, 1, 1},
	}
	freq, time := smoothingFilterAxes(1, 2)
	kernel := smoothingFilter(1, 2)
	gp := gateParams{
		smoothMask:       true,
		smoothKernel:     kernel,
		smoothKernelFreq: freq,
		smoothKernelTime: time,
	}
	got := smoothMaskSame(mask, gp)
	want := directConv2DSame(mask, kernel)

	for i := range want {
		for j := range want[i] {
			assert.InDeltaf(t, want[i][j], got[i][j], 1e-12, "[%d,%d]", i, j)
		}
	}
}

func TestResolveGateParamsZeroSmoothAxisIsOneBin(t *testing.T) {
	opt := Options{NFFT: 8, WinLength: 8, HopLength: 2, FreqMaskSmoothHz: 0, TimeMaskSmoothMs: 500}
	gp, err := resolveGateParams(opt, 8)
	require.NoError(t, err)
	require.True(t, gp.smoothMask, "expected time-only smoothing to be enabled")
	require.Len(t, gp.smoothKernel, 1, "freq kernel rows when FreqMaskSmoothHz=0")
	require.Len(t, gp.smoothKernel[0], 5, "time kernel cols")

	opt = Options{NFFT: 8, WinLength: 8, HopLength: 2, FreqMaskSmoothHz: 4, TimeMaskSmoothMs: 0}
	gp, err = resolveGateParams(opt, 8)
	require.NoError(t, err)
	require.True(t, gp.smoothMask, "expected freq-only smoothing to be enabled")
	require.Len(t, gp.smoothKernel, 5, "freq kernel rows")
	require.Len(t, gp.smoothKernel[0], 1, "time kernel cols when TimeMaskSmoothMs=0")
}

func TestFinishStationaryMaskBlendsBeforeSmoothing(t *testing.T) {
	mask := [][]float64{{0, 1, 0}}
	gp := gateParams{
		smoothMask:   true,
		smoothKernel: smoothingFilter(0, 1),
	}
	got := finishStationaryMask(mask, gp, 0.5)
	want := [][]float64{{0.5, 0.75, 0.5}}

	for i := range want {
		for j := range want[i] {
			assert.InDeltaf(t, want[i][j], got[i][j], 1e-12, "[%d,%d]", i, j)
		}
	}
}

func TestFinishStationaryMaskPropDecreaseZeroStillSmoothsEdges(t *testing.T) {
	mask := [][]float64{{0, 0, 0}}
	gp := gateParams{
		smoothMask:   true,
		smoothKernel: smoothingFilter(0, 1),
	}
	got := finishStationaryMask(mask, gp, 0)
	want := [][]float64{{0.75, 1, 0.75}}

	for i := range want {
		for j := range want[i] {
			assert.InDeltaf(t, want[i][j], got[i][j], 1e-12, "[%d,%d]", i, j)
		}
	}
}

func TestReadChunkPadding(t *testing.T) {
	y := []float64{1, 2, 3, 4, 5}
	c := readChunk(y, -2, 7)
	want := []float64{0, 0, 1, 2, 3, 4, 5, 0, 0}
	require.Len(t, c, len(want))
	for i := range want {
		assert.Equalf(t, want[i], c[i], "[%d]", i)
	}
}

// TestApplyDefaultsPreservesChunkSizeZero pins down "ChunkSize=0
// disables chunking" through applyDefaults. With an input larger than
// the old default (600000), the chunk filter must run exactly once
// after defaulting. Under the previous applyDefaults bug, ChunkSize=0
// was rewritten to 600000 and the filter ran ceil(n/600000) times.
func TestApplyDefaultsPreservesChunkSizeZero(t *testing.T) {
	const n = 700_000 // > old default ChunkSize of 600_000

	opt := DefaultOptions()
	opt.ChunkSize = 0 // request unchunked
	opt.applyDefaults()

	require.Zero(t, opt.ChunkSize, "applyDefaults rewrote ChunkSize=0")

	var calls atomic.Int32
	filter := func(c []float64) []float64 {
		calls.Add(1)
		out := make([]float64, len(c))
		copy(out, c)
		return out
	}
	y := make([]float64, n)
	out := runChunked(y, opt.ChunkSize, opt.Padding, opt.NJobs, filter)
	assert.Equal(t, int32(1), calls.Load(), "ChunkSize=0 must run exactly once regardless of input length")
	require.Len(t, out, n)
}

func TestRunChunkedIdentity(t *testing.T) {
	// Identity filter: chunked or not, output should match input.
	y := make([]float64, 1000)
	for i := range y {
		y[i] = float64(i) * 0.1
	}
	identity := func(c []float64) []float64 {
		o := make([]float64, len(c))
		copy(o, c)
		return o
	}
	for _, cs := range []int{0, 200, 137, 1} {
		out := runChunked(y, cs, 50, 4, identity)
		require.Len(t, out, len(y), "chunkSize=%d", cs)
		for i := range y {
			assert.Equalf(t, y[i], out[i], "chunkSize=%d [%d]", cs, i)
		}
	}
}

// TestRunChunkedNJobsZeroParallelises pins down the documented
// "NJobs == 0 => GOMAXPROCS" semantics. If runChunked silently treats
// 0 as 1, callers inheriting the default from DefaultOptions() lose
// parallelism.
func TestRunChunkedNJobsZeroParallelises(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("need GOMAXPROCS >= 2")
	}
	const chunks = 8
	const chunkSize = 64
	y := make([]float64, chunks*chunkSize)

	var inFlight, peak int32
	var mu sync.Mutex
	cond := sync.NewCond(&mu)
	started := 0
	filter := func(c []float64) []float64 {
		cur := atomic.AddInt32(&inFlight, 1)
		for {
			p := atomic.LoadInt32(&peak)
			if cur <= p || atomic.CompareAndSwapInt32(&peak, p, cur) {
				break
			}
		}
		// Wait until we know more than one job has actually started in
		// parallel before any job is allowed to finish.
		mu.Lock()
		started++
		if started >= 2 {
			cond.Broadcast()
		}
		for started < 2 {
			cond.Wait()
		}
		mu.Unlock()
		atomic.AddInt32(&inFlight, -1)
		out := make([]float64, len(c))
		copy(out, c)
		return out
	}
	out := runChunked(y, chunkSize, 0, 0 /* NJobs=0 => GOMAXPROCS */, filter)
	require.Len(t, out, len(y))
	require.GreaterOrEqualf(t, peak, int32(2), "NJobs=0 should resolve to GOMAXPROCS=%d", runtime.GOMAXPROCS(0))
}
