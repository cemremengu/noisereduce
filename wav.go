package noisereduce

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Minimal RIFF/WAVE reader and writer for the formats used by the
// noisereduce test fixtures (16-bit PCM and 32-bit float). Stdlib only.

// ReadWAV reads a WAVE file and returns samples shaped [channels][frames]
// as float64 in [-1, 1] (PCM normalised by 32768).
func ReadWAV(path string) (samples [][]float64, sampleRate int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	return readWAV(f)
}

// ReadWAVBytes reads WAVE data from memory and returns samples shaped
// [channels][frames] as float64 in [-1, 1] (PCM normalised by 32768).
func ReadWAVBytes(data []byte) (samples [][]float64, sampleRate int, err error) {
	return readWAV(bytes.NewReader(data))
}

type wavFmt struct {
	audioFormat   uint16
	numChannels   uint16
	sampleRate    uint32
	byteRate      uint32
	blockAlign    uint16
	bitsPerSample uint16
}

func readWAV(r io.Reader) ([][]float64, int, error) {
	buf, err := io.ReadAll(r)
	if err != nil {
		return nil, 0, err
	}
	if len(buf) < 44 {
		return nil, 0, errors.New("noisereduce: WAV too short")
	}
	if string(buf[0:4]) != "RIFF" || string(buf[8:12]) != "WAVE" {
		return nil, 0, errors.New("noisereduce: not a RIFF/WAVE file")
	}

	var fmtChunk wavFmt
	var dataStart int
	var dataSize int
	off := 12
	haveFmt := false
	for off+8 <= len(buf) {
		id := string(buf[off : off+4])
		size := int(binary.LittleEndian.Uint32(buf[off+4 : off+8]))
		body := off + 8
		if body+size > len(buf) {
			return nil, 0, fmt.Errorf("noisereduce: chunk %q overruns file", id)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, 0, errors.New("noisereduce: fmt chunk too small")
			}
			fmtChunk.audioFormat = binary.LittleEndian.Uint16(buf[body : body+2])
			fmtChunk.numChannels = binary.LittleEndian.Uint16(buf[body+2 : body+4])
			fmtChunk.sampleRate = binary.LittleEndian.Uint32(buf[body+4 : body+8])
			fmtChunk.byteRate = binary.LittleEndian.Uint32(buf[body+8 : body+12])
			fmtChunk.blockAlign = binary.LittleEndian.Uint16(buf[body+12 : body+14])
			fmtChunk.bitsPerSample = binary.LittleEndian.Uint16(buf[body+14 : body+16])
			haveFmt = true
		case "data":
			dataStart = body
			dataSize = size
		}
		off = body + size
		if size%2 == 1 {
			off++ // RIFF chunks pad to even.
		}
		if haveFmt && dataStart != 0 {
			break
		}
	}
	if !haveFmt {
		return nil, 0, errors.New("noisereduce: missing fmt chunk")
	}
	if dataStart == 0 {
		return nil, 0, errors.New("noisereduce: missing data chunk")
	}

	return decodeWAVData(buf[dataStart:dataStart+dataSize], fmtChunk, dataSize)
}

func decodeWAVData(data []byte, fmtChunk wavFmt, dataSize int) ([][]float64, int, error) {
	channels := int(fmtChunk.numChannels)
	if channels == 0 {
		return nil, 0, errors.New("noisereduce: zero channels")
	}
	bitsPerSample := int(fmtChunk.bitsPerSample)
	bytesPerSample := bitsPerSample / 8
	frame := channels * bytesPerSample
	if frame == 0 {
		return nil, 0, errors.New("noisereduce: invalid frame size")
	}
	frames := dataSize / frame
	out := make([][]float64, channels)
	for c := range out {
		out[c] = make([]float64, frames)
	}

	switch fmtChunk.audioFormat {
	case 1: // PCM
		switch bitsPerSample {
		case 16:
			scale := 1.0 / 32768.0
			idx := 0
			for f := 0; f < frames; f++ {
				for c := 0; c < channels; c++ {
					s := int16(data[idx]) | int16(data[idx+1])<<8
					out[c][f] = float64(s) * scale
					idx += 2
				}
			}
		case 24:
			scale := 1.0 / 8388608.0
			idx := 0
			for f := 0; f < frames; f++ {
				for c := 0; c < channels; c++ {
					v := int32(data[idx]) | int32(data[idx+1])<<8 | int32(data[idx+2])<<16
					if data[idx+2]&0x80 != 0 {
						v |= ^int32(0xFFFFFF)
					}
					out[c][f] = float64(v) * scale
					idx += 3
				}
			}
		case 32:
			scale := 1.0 / 2147483648.0
			idx := 0
			for f := 0; f < frames; f++ {
				for c := 0; c < channels; c++ {
					v := int32(data[idx]) |
						int32(data[idx+1])<<8 |
						int32(data[idx+2])<<16 |
						int32(data[idx+3])<<24
					out[c][f] = float64(v) * scale
					idx += 4
				}
			}
		default:
			return nil, 0, fmt.Errorf("noisereduce: unsupported PCM bit depth %d", bitsPerSample)
		}
	case 3: // IEEE float
		switch bitsPerSample {
		case 32:
			idx := 0
			for f := 0; f < frames; f++ {
				for c := 0; c < channels; c++ {
					bits := binary.LittleEndian.Uint32(data[idx : idx+4])
					out[c][f] = float64(math.Float32frombits(bits))
					idx += 4
				}
			}
		case 64:
			idx := 0
			for f := 0; f < frames; f++ {
				for c := 0; c < channels; c++ {
					bits := binary.LittleEndian.Uint64(data[idx : idx+8])
					out[c][f] = math.Float64frombits(bits)
					idx += 8
				}
			}
		default:
			return nil, 0, fmt.Errorf("noisereduce: unsupported float bit depth %d", bitsPerSample)
		}
	default:
		return nil, 0, fmt.Errorf("noisereduce: unsupported audio format %d", fmtChunk.audioFormat)
	}

	return out, int(fmtChunk.sampleRate), nil
}

// ReduceNoiseWAVBytes reads WAV data from memory, applies ReduceNoise, and
// returns a 16-bit PCM WAV byte slice.
func ReduceNoiseWAVBytes(data []byte, opt Options) ([]byte, error) {
	samples, sr, err := ReadWAVBytes(data)
	if err != nil {
		return nil, err
	}
	reduced, err := ReduceNoise(samples, sr, opt)
	if err != nil {
		return nil, err
	}
	return WriteWAVPCM16Bytes(reduced, sr)
}

// WriteWAVPCM16Bytes encodes [channels][frames] float64 samples as a 16-bit
// PCM WAV byte slice. Samples are clamped to [-1, 1] and scaled by 32767.
func WriteWAVPCM16Bytes(samples [][]float64, sampleRate int) ([]byte, error) {
	return encodeWAVPCM16(samples, sampleRate)
}

// WriteWAVPCM16 writes [channels][frames] float64 samples as 16-bit PCM.
// Samples are clamped to [-1, 1] and scaled by 32767.
func WriteWAVPCM16(path string, samples [][]float64, sampleRate int) error {
	data, err := encodeWAVPCM16(samples, sampleRate)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func encodeWAVPCM16(samples [][]float64, sampleRate int) ([]byte, error) {
	if len(samples) == 0 {
		return nil, errors.New("noisereduce: no channels")
	}
	if len(samples) > 65535 {
		return nil, errors.New("noisereduce: too many channels")
	}
	if sampleRate <= 0 {
		return nil, errors.New("noisereduce: sample rate must be positive")
	}
	channels := len(samples)
	if channels > math.MaxUint16 {
		return nil, errors.New("noisereduce: too many channels")
	}
	blockAlign := channels * 2
	if blockAlign > math.MaxUint16 {
		return nil, errors.New("noisereduce: WAV block align too large")
	}
	if sampleRate > math.MaxUint32/blockAlign {
		return nil, errors.New("noisereduce: WAV byte rate too large")
	}
	byteRate := sampleRate * blockAlign
	channels16 := uint16(channels)     // #nosec G115
	sampleRate32 := uint32(sampleRate) // #nosec G115
	byteRate32 := uint32(byteRate)     // #nosec G115
	blockAlign16 := uint16(blockAlign) // #nosec G115
	frames := len(samples[0])
	for c := 1; c < channels; c++ {
		if len(samples[c]) != frames {
			return nil, errors.New("noisereduce: channel length mismatch")
		}
	}

	dataBytes := frames * channels * 2
	if dataBytes > int(^uint32(0))-36 {
		return nil, errors.New("noisereduce: WAV data too large")
	}

	var buf bytes.Buffer
	buf.Grow(44 + dataBytes)

	buf.WriteString("RIFF")
	if err := writeWAVValue(&buf, uint32(36+dataBytes)); err != nil {
		return nil, err
	}
	buf.WriteString("WAVE")

	buf.WriteString("fmt ")
	if err := writeWAVValue(&buf, uint32(16)); err != nil {
		return nil, err
	}
	if err := writeWAVValue(&buf, uint16(1)); err != nil { // PCM
		return nil, err
	}
	if err := writeWAVValue(&buf, channels16); err != nil {
		return nil, err
	}
	if err := writeWAVValue(&buf, sampleRate32); err != nil {
		return nil, err
	}
	if err := writeWAVValue(&buf, byteRate32); err != nil {
		return nil, err
	}
	if err := writeWAVValue(&buf, blockAlign16); err != nil {
		return nil, err
	}
	if err := writeWAVValue(&buf, uint16(16)); err != nil {
		return nil, err
	}

	buf.WriteString("data")
	if err := writeWAVValue(&buf, uint32(dataBytes)); err != nil {
		return nil, err
	}

	for f := 0; f < frames; f++ {
		for c := 0; c < channels; c++ {
			v := samples[c][f]
			if v > 1 {
				v = 1
			} else if v < -1 {
				v = -1
			}
			s := int16(math.Round(v * 32767))
			if err := writeWAVValue(&buf, s); err != nil {
				return nil, err
			}
		}
	}

	return buf.Bytes(), nil
}

func writeWAVValue(buf *bytes.Buffer, v any) error {
	return binary.Write(buf, binary.LittleEndian, v)
}
