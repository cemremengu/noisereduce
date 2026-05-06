package noisereduce

func runNonStationary(y [][]float64, sr int, opt Options) ([][]float64, error) {
	gp, err := resolveGateParams(opt, sr)
	if err != nil {
		return nil, err
	}

	b := timeSmoothCoefficient(opt.TimeConstantS, sr, gp.hop)

	filter := func(chunk []float64) []float64 {
		z := stft(chunk, gp.nfft, gp.winLen, gp.hop)
		absZ := magnitude(z)

		// Time-smoothed magnitude per frequency row (filtfilt along axis=-1).
		smooth := make([][]float64, len(absZ))
		for i, row := range absZ {
			smooth[i] = filtfilt1Pole(b, row)
		}

		// (|X| - smooth) / smooth, then sigmoid.
		mask := make([][]float64, len(absZ))
		for i := range absZ {
			row := make([]float64, len(absZ[i]))
			for j := range absZ[i] {
				s := smooth[i][j]
				if s == 0 {
					row[j] = sigmoid(0, -opt.ThreshNMultNonstationary, opt.SigmoidSlopeNonstationary)
					continue
				}
				ratio := (absZ[i][j] - s) / s
				row[j] = sigmoid(ratio, -opt.ThreshNMultNonstationary, opt.SigmoidSlopeNonstationary)
			}
			mask[i] = row
		}

		if gp.smoothMask {
			mask = smoothMaskSame(mask, gp)
		}
		applyPropDecrease(mask, opt.PropDecrease)
		applyMask(z, mask)
		return istft(z, gp.nfft, gp.winLen, gp.hop, len(chunk))
	}

	out := make([][]float64, len(y))
	for ci := range y {
		out[ci] = runChunked(y[ci], opt.ChunkSize, opt.Padding, opt.NJobs, filter)
	}
	return out, nil
}
