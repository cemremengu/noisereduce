package noisereduce

import "fmt"

func runStationary(y [][]float64, sr int, opt Options) ([][]float64, error) {
	gp, err := resolveGateParams(opt, sr)
	if err != nil {
		return nil, err
	}

	// Build the noise reference by averaging across channels.
	noiseSrc := opt.YNoise
	if noiseSrc == nil {
		noiseSrc = y
	} else if err := validateYNoise(noiseSrc); err != nil {
		return nil, err
	}
	noise := meanAcrossChannels(noiseSrc)
	if opt.ClipNoiseStationary && len(noise) > opt.ChunkSize && opt.ChunkSize > 0 {
		noise = noise[:opt.ChunkSize]
	}

	// Per-frequency-bin mean+std of dB magnitude (axis=1).
	noiseSTFT := stft(noise, gp.nfft, gp.winLen, gp.hop)
	noiseDB := amplitudeToDB(noiseSTFT, 80)
	meanNoise := rowMean(noiseDB)
	stdNoise := rowStd(noiseDB)
	thresh := make([]float64, len(meanNoise))
	for i := range thresh {
		thresh[i] = meanNoise[i] + stdNoise[i]*opt.NStdThreshStationary
	}

	filter := func(chunk []float64) []float64 {
		z := stft(chunk, gp.nfft, gp.winLen, gp.hop)
		zDB := amplitudeToDB(z, 80)
		mask := make([][]float64, len(zDB))
		for i, row := range zDB {
			m := make([]float64, len(row))
			for j, v := range row {
				if v > thresh[i] {
					m[j] = 1
				}
			}
			mask[i] = m
		}
		mask = finishStationaryMask(mask, gp, opt.PropDecrease)
		applyMask(z, mask)
		return istft(z, gp.nfft, gp.winLen, gp.hop, len(chunk))
	}

	out := make([][]float64, len(y))
	for ci := range y {
		out[ci] = runChunked(y[ci], opt.ChunkSize, opt.Padding, opt.NJobs, filter)
	}
	return out, nil
}

func finishStationaryMask(mask [][]float64, gp gateParams, propDecrease float64) [][]float64 {
	applyPropDecrease(mask, propDecrease)
	if gp.smoothMask {
		mask = smoothMaskSame(mask, gp)
	}
	return mask
}

func validateYNoise(y [][]float64) error {
	if len(y) == 0 {
		return nil
	}
	frames := len(y[0])
	for i, ch := range y {
		if len(ch) != frames {
			return fmt.Errorf("noisereduce: YNoise channel %d has %d frames, expected %d", i, len(ch), frames)
		}
	}
	return nil
}

// meanAcrossChannels collapses a [channels][frames] signal to a single
// channel by averaging across channels.
func meanAcrossChannels(y [][]float64) []float64 {
	if len(y) == 0 {
		return nil
	}
	if len(y) == 1 {
		out := make([]float64, len(y[0]))
		copy(out, y[0])
		return out
	}
	frames := len(y[0])
	out := make([]float64, frames)
	for _, ch := range y {
		for i, v := range ch {
			out[i] += v
		}
	}
	inv := 1.0 / float64(len(y))
	for i := range out {
		out[i] *= inv
	}
	return out
}
