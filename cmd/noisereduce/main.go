// Command noisereduce applies spectral-gating noise reduction to a WAV file.
//
//	noisereduce -in input.wav -out output.wav
//	            [-stationary] [-prop-decrease 1.0]
//	            [-n-fft 1024] [-time-constant 2.0] [-jobs 0]
//	            [-noise noise.wav]
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/cemremengu/noisereduce"
)

func main() {
	in := flag.String("in", "", "input WAV file (required)")
	out := flag.String("out", "", "output WAV file (required, written as 16-bit PCM)")
	noisePath := flag.String("noise", "", "optional noise reference WAV (stationary only)")
	stationary := flag.Bool("stationary", false, "use stationary spectral gating")
	prop := flag.Float64("prop-decrease", 1.0, "proportion of noise to suppress (0..1)")
	nfft := flag.Int("n-fft", 1024, "FFT size")
	winLen := flag.Int("win-length", 0, "STFT window length (0 = n-fft)")
	hopLen := flag.Int("hop-length", 0, "STFT hop length (0 = win-length/4)")
	timeConst := flag.Float64("time-constant", 2.0, "non-stationary IIR time constant in seconds")
	threshN := flag.Float64("n-std-thresh", 1.5, "stationary stddev multiplier")
	chunkSize := flag.Int("chunk-size", 600000, "frames per chunk; 0 disables chunking")
	padding := flag.Int("padding", 30000, "padding around each chunk")
	jobs := flag.Int("jobs", 0, "parallel chunk workers (0 = GOMAXPROCS)")
	flag.Parse()

	if *in == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *noisePath != "" && !*stationary {
		log.Fatalf("-noise requires -stationary; non-stationary mode does not use a noise reference")
	}

	y, sr, err := noisereduce.ReadWAV(*in)
	if err != nil {
		log.Fatalf("read %s: %v", *in, err)
	}

	opt := noisereduce.DefaultOptions()
	opt.PropDecrease = *prop
	opt.NFFT = *nfft
	opt.WinLength = *winLen
	opt.HopLength = *hopLen
	opt.TimeConstantS = *timeConst
	opt.NStdThreshStationary = *threshN
	opt.ChunkSize = *chunkSize
	opt.Padding = *padding
	opt.NJobs = *jobs
	if *stationary {
		opt.Algorithm = noisereduce.Stationary
	}

	if *noisePath != "" {
		ny, nsr, err := noisereduce.ReadWAV(*noisePath)
		if err != nil {
			log.Fatalf("read noise %s: %v", *noisePath, err)
		}
		if nsr != sr {
			// Reusing a noise reference at a different sample rate would
			// shift every FFT bin's physical frequency, so the stationary
			// thresholds would gate the wrong bands. Reject up front
			// rather than silently producing wrong output.
			log.Fatalf("noise sample rate (%d Hz) does not match input (%d Hz); "+
				"resample %s before passing it to -noise", nsr, sr, *noisePath)
		}
		opt.YNoise = ny
	}

	reduced, err := noisereduce.ReduceNoise(y, sr, opt)
	if err != nil {
		log.Fatalf("reduce: %v", err)
	}

	if err := noisereduce.WriteWAVPCM16(*out, reduced, sr); err != nil {
		log.Fatalf("write %s: %v", *out, err)
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%d samples, %d Hz, %d ch)\n",
		*out, len(reduced[0]), sr, len(reduced))
}
