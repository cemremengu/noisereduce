# noisereduce

Pure-Go spectral-gating noise reduction for audio-rate signals. This project is a Go port of the Python [`noisereduce` library](https://pypi.org/project/noisereduce/3.0.3/). The package supports stationary and non-stationary noise reduction, includes a small WAV reader/writer, and ships a CLI for processing WAV files. Please refer to the original Python project for more details on the underlying algorithms and their parameters.

## Features

- Pure Go library API for mono or multichannel signals shaped `[channels][frames]float64`.
- In-memory WAV byte input/output for services and pipelines that do not use files.
- Non-stationary spectral gating with a continuously estimated noise floor.
- Stationary spectral gating with an optional noise reference clip.
- Chunked processing with parallel workers for long recordings.
- WAV input support for PCM and IEEE float data, with 16-bit PCM WAV output.
- No cgo dependency.

## Install

```sh
go get github.com/cemremengu/noisereduce
```

## CLI Usage

Run the CLI from the repository:

```sh
go run ./cmd/noisereduce -in noisy.wav -out clean.wav
```

Use stationary mode with a separate noise reference:

```sh
go run ./cmd/noisereduce \
  -in noisy.wav \
  -out clean.wav \
  -stationary \
  -noise sample_noise.wav
```

Build and install the command:

```sh
go install github.com/cemremengu/noisereduce/cmd/noisereduce@latest
noisereduce -in noisy.wav -out clean.wav
```

Common CLI flags:

| Flag | Description | Default |
| --- | --- | --- |
| `-in` | Input WAV path. Required. | |
| `-out` | Output WAV path. Required. Writes 16-bit PCM. | |
| `-stationary` | Use stationary gating instead of non-stationary gating. | `false` |
| `-noise` | Noise reference WAV for stationary mode. | |
| `-prop-decrease` | Proportion of noise to suppress, from `0` to `1`. | `1.0` |
| `-n-fft` | FFT size. | `1024` |
| `-win-length` | STFT window length. `0` means `n-fft`. | `0` |
| `-hop-length` | STFT hop length. `0` means `win-length/4`. | `0` |
| `-time-constant` | Non-stationary IIR time constant in seconds. | `2.0` |
| `-n-std-thresh` | Stationary standard-deviation threshold multiplier. | `1.5` |
| `-chunk-size` | Frames per chunk. `0` disables chunking. | `600000` |
| `-padding` | Frames of padding around each chunk. | `30000` |
| `-jobs` | Parallel chunk workers. `0` uses `GOMAXPROCS`. | `0` |

## Library Usage

Read a WAV, reduce noise with the default non-stationary gate, and write a new WAV:

```go
package main

import (
    "log"

    noisereduce "github.com/cemremengu/noisereduce"
)

func main() {
    samples, sr, err := noisereduce.ReadWAV("noisy.wav")
    if err != nil {
        log.Fatal(err)
    }

    opt := noisereduce.DefaultOptions()
    opt.Algorithm = noisereduce.NonStationary
    opt.PropDecrease = 1.0

    denoised, err := noisereduce.ReduceNoise(samples, sr, opt)
    if err != nil {
        log.Fatal(err)
    }

    if err := noisereduce.WriteWAVPCM16("clean.wav", denoised, sr); err != nil {
        log.Fatal(err)
    }
}
```

Use stationary mode with an explicit noise clip:

```go
samples, sr, err := noisereduce.ReadWAV("noisy.wav")
if err != nil {
    log.Fatal(err)
}

noise, noiseSR, err := noisereduce.ReadWAV("noise.wav")
if err != nil {
    log.Fatal(err)
}
if noiseSR != sr {
    log.Fatalf("noise sample rate %d does not match input sample rate %d", noiseSR, sr)
}

opt := noisereduce.DefaultOptions()
opt.Algorithm = noisereduce.Stationary
opt.YNoise = noise
opt.NStdThreshStationary = 1.5
opt.NJobs = 0

clean, err := noisereduce.ReduceNoise(samples, sr, opt)
if err != nil {
    log.Fatal(err)
}
```

Use the mono helper when your data is a single `[]float64` channel:

```go
cleanMono, err := noisereduce.ReduceNoiseMono(noisyMono, sampleRate, noisereduce.DefaultOptions())
if err != nil {
    log.Fatal(err)
}
```

Process WAV data already held in memory:

```go
cleanWAV, err := noisereduce.ReduceNoiseWAVBytes(noisyWAV, noisereduce.DefaultOptions())
if err != nil {
    log.Fatal(err)
}
```

Or decode and encode WAV bytes explicitly:

```go
samples, sr, err := noisereduce.ReadWAVBytes(noisyWAV)
if err != nil {
    log.Fatal(err)
}

clean, err := noisereduce.ReduceNoise(samples, sr, noisereduce.DefaultOptions())
if err != nil {
    log.Fatal(err)
}

cleanWAV, err := noisereduce.WriteWAVPCM16Bytes(clean, sr)
if err != nil {
    log.Fatal(err)
}
```

## Options

Start with `DefaultOptions()` and override only the fields you need.

```go
opt := noisereduce.DefaultOptions()
opt.Algorithm = noisereduce.NonStationary
opt.PropDecrease = 0.8
opt.TimeConstantS = 1.5
opt.FreqMaskSmoothHz = 500
opt.TimeMaskSmoothMs = 50
opt.ChunkSize = 0 // one-shot processing
```

Important fields:

| Field | Description |
| --- | --- |
| `Algorithm` | `NonStationary` or `Stationary`. |
| `YNoise` | Optional stationary-mode noise reference shaped `[channels][frames]`. |
| `PropDecrease` | Blend amount for suppression. `0` preserves the signal, `1` applies full suppression. |
| `TimeConstantS` | Time constant for non-stationary floor estimation. |
| `FreqMaskSmoothHz` | Frequency smoothing width for the spectral mask. `0` disables frequency smoothing. |
| `TimeMaskSmoothMs` | Time smoothing width for the spectral mask. `0` disables time smoothing. |
| `ThreshNMultNonstationary` | Non-stationary sigmoid shift. |
| `SigmoidSlopeNonstationary` | Non-stationary sigmoid slope. |
| `NStdThreshStationary` | Stationary threshold multiplier. |
| `NFFT`, `WinLength`, `HopLength` | STFT geometry. |
| `ChunkSize`, `Padding`, `NJobs` | Chunking and parallelism controls. |

## Choosing An Algorithm

Stationary gating is useful when you have a representative noise-only clip or when the noise floor is stable across the recording. If `YNoise` is not supplied, the input signal is also used as the noise reference.

Non-stationary gating is useful when background noise changes over time. It estimates a time-smoothed noise floor from the input itself and does not use a separate noise clip.

## Development

Run tests:

```sh
go test ./...
```

Run the noise-reduction benchmarks:

```sh
go test -run '^$' -bench 'BenchmarkReduceNoise' -benchmem
```

Capture a CPU profile while running the same benchmarks:

```sh
go test -run '^$' -bench 'BenchmarkReduceNoise' -benchmem -cpuprofile cpu.out
```

Build the CLI:

```sh
go build ./cmd/noisereduce
```

## Disclaimer

This library was built with significant help from Claude and Codex and may include inaccuracies, inefficient code patterns, or potential security vulnerabilities. Please use it with caution. If you encounter any issues, feel free to open an issue or submit a pull request.
