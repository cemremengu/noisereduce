// Package noisereduce performs spectral-gating noise reduction on
// audio-rate signals. It is a pure-Go implementation with two algorithms:
//
//   - Stationary spectral gating (NStdThreshStationary, optional noise
//     clip). Statistics are computed once over a noise sample (or the
//     signal itself) and a fixed threshold is applied to the entire
//     input.
//   - Non-stationary spectral gating, where the noise floor is estimated
//     continuously from a time-smoothed magnitude spectrogram.
package noisereduce
