package noisereduce

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWAVBytesRoundTripPCM16(t *testing.T) {
	samples := [][]float64{
		{0, 0.25, -0.5, 1, -1},
		{0.5, -0.25, 0.125, -1, 1},
	}

	data, err := WriteWAVPCM16Bytes(samples, 16000)
	require.NoError(t, err)
	require.Len(t, data, 44+len(samples)*len(samples[0])*2)

	got, sr, err := ReadWAVBytes(data)
	require.NoError(t, err)
	require.Equal(t, 16000, sr)
	require.Len(t, got, len(samples))
	require.Len(t, got[0], len(samples[0]))

	for c := range samples {
		for f := range samples[c] {
			assert.InDeltaf(t, samples[c][f], got[c][f], 1.0/32768.0, "channel %d frame %d", c, f)
		}
	}
}

func TestReduceNoiseWAVBytesReturnsValidWAV(t *testing.T) {
	samples := [][]float64{make([]float64, 2048)}
	for i := range samples[0] {
		if i%2 == 0 {
			samples[0][i] = 0.2
		} else {
			samples[0][i] = -0.2
		}
	}
	data, err := WriteWAVPCM16Bytes(samples, 22050)
	require.NoError(t, err)

	opt := DefaultOptions()
	opt.PropDecrease = 0
	opt.ChunkSize = 0
	outData, err := ReduceNoiseWAVBytes(data, opt)
	require.NoError(t, err)

	out, sr, err := ReadWAVBytes(outData)
	require.NoError(t, err)
	require.Equal(t, 22050, sr)
	require.Len(t, out, 1)
	require.Len(t, out[0], len(samples[0]))
}

func TestReadWAVBytesRejectsInvalidData(t *testing.T) {
	_, _, err := ReadWAVBytes([]byte("not wav"))
	require.Error(t, err)
}

func TestWriteWAVPCM16BytesRejectsInvalidInput(t *testing.T) {
	_, err := WriteWAVPCM16Bytes(nil, 16000)
	require.Error(t, err)

	_, err = WriteWAVPCM16Bytes([][]float64{{0}, {0, 1}}, 16000)
	require.Error(t, err)

	_, err = WriteWAVPCM16Bytes([][]float64{{0}}, 0)
	require.Error(t, err)
}
