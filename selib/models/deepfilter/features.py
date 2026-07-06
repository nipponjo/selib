"""DeepFilterNet feature extraction and synthesis helpers.

The functions in this module mirror the small part of libDF needed for offline
ONNX inference: ERB feature construction, normalization, the libDF-compatible
streaming STFT, and inverse synthesis.

When numba is installed, the sequential normalization and overlap-add loops are
compiled automatically. NumPy remains the fallback and is used for FFTs and
matrix operations.
"""

import math
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba installed.
    njit = None

NUMBA_AVAILABLE = njit is not None


if NUMBA_AVAILABLE:
    @njit
    def _erb_norm_numba(x: np.ndarray, alpha: float) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        n_bands = x.shape[1]
        state = np.empty(n_bands, dtype=np.float32)
        for band in range(n_bands):
            if n_bands == 1:
                state[band] = -60.0
            else:
                state[band] = -60.0 + (-30.0 * band / (n_bands - 1))
        for frame in range(x.shape[0]):
            for band in range(n_bands):
                state[band] = x[frame, band] * (1.0 - alpha) + state[band] * alpha
                out[frame, band] = (x[frame, band] - state[band]) / 40.0
        return out


    @njit
    def _unit_norm_numba(spec: np.ndarray, alpha: float, eps: float) -> np.ndarray:
        out = np.empty_like(spec, dtype=np.complex64)
        n_bins = spec.shape[1]
        state = np.empty(n_bins, dtype=np.float32)
        for bin_idx in range(n_bins):
            if n_bins == 1:
                state[bin_idx] = 0.001
            else:
                state[bin_idx] = 0.001 + (-0.0009 * bin_idx / (n_bins - 1))
        for frame in range(spec.shape[0]):
            for bin_idx in range(n_bins):
                value = spec[frame, bin_idx]
                mag = math.sqrt(value.real * value.real + value.imag * value.imag)
                state[bin_idx] = mag * (1.0 - alpha) + state[bin_idx] * alpha
                denom = math.sqrt(max(state[bin_idx], eps))
                out[frame, bin_idx] = value / denom
        return out


    @njit
    def _analysis_frames_numba(wave: np.ndarray,
                               window: np.ndarray,
                               n_fft: int,
                               hop_length: int,
                               wnorm: float) -> np.ndarray:
        n_frames = wave.shape[0] // hop_length
        mem_len = n_fft - hop_length
        analysis_split = mem_len - hop_length
        analysis_mem = np.zeros(mem_len, dtype=np.float32)
        frames = np.empty((n_frames, n_fft), dtype=np.float32)

        for frame_idx in range(n_frames):
            start = frame_idx * hop_length
            for sample in range(mem_len):
                frames[frame_idx, sample] = analysis_mem[sample] * window[sample]
            for sample in range(hop_length):
                frames[frame_idx, mem_len + sample] = (
                    wave[start + sample] * window[mem_len + sample] * wnorm
                )

            if analysis_split > 0:
                for sample in range(analysis_split):
                    analysis_mem[sample] = analysis_mem[sample + hop_length]
            for sample in range(hop_length):
                analysis_mem[analysis_split + sample] = wave[start + sample]

            for sample in range(mem_len):
                frames[frame_idx, sample] *= wnorm
        return frames


    @njit
    def _synthesis_overlap_add_numba(frames_td: np.ndarray,
                                     hop_length: int) -> np.ndarray:
        n_frames, n_fft = frames_td.shape
        mem_len = n_fft - hop_length
        split = mem_len - hop_length
        synthesis_mem = np.zeros(mem_len, dtype=np.float32)
        audio = np.empty(n_frames * hop_length, dtype=np.float32)

        for frame_idx in range(n_frames):
            out_start = frame_idx * hop_length
            for sample in range(hop_length):
                audio[out_start + sample] = frames_td[frame_idx, sample] + synthesis_mem[sample]

            if split > 0:
                for sample in range(split):
                    synthesis_mem[sample] = synthesis_mem[sample + hop_length]
                for sample in range(split):
                    synthesis_mem[sample] += frames_td[frame_idx, hop_length + sample]
            for sample in range(mem_len - split):
                synthesis_mem[split + sample] = frames_td[frame_idx, hop_length + split + sample]

        return audio


def calculate_norm_alpha(sr: int, hop_size: int, tau: float) -> float:
    """Calculate the rounded exponential smoothing factor used by libDF."""
    alpha = math.exp(-(hop_size / sr) / tau)
    precision = 3
    rounded = 1.0
    while rounded >= 1.0:
        rounded = round(alpha, precision)
        precision += 1
    return rounded


def erb_filterbank_from_widths(widths: np.ndarray,
                               normalized: bool = True,
                               inverse: bool = False) -> np.ndarray:
    """Build a rectangular ERB filterbank from per-band bin counts."""
    n_freqs = int(np.sum(widths))
    starts = np.cumsum(np.concatenate(([0], widths.astype(np.int64))))[:-1]
    fb = np.zeros((n_freqs, len(widths)), dtype=np.float32)
    for band, (start, width) in enumerate(zip(starts, widths.astype(np.int64))):
        fb[start:start + width, band] = 1.0
    if inverse:
        fb = fb.T
        if not normalized:
            denom = np.maximum(fb.sum(axis=1, keepdims=True), 1e-12)
            fb = fb / denom
    elif normalized:
        denom = np.maximum(fb.sum(axis=0, keepdims=True), 1e-12)
        fb = fb / denom
    return fb.astype(np.float32)


def erb_norm(x: np.ndarray,
             alpha: float) -> np.ndarray:
    """DeepFilterNet ERB mean normalization.

    Matches libDF band_mean_norm_erb:
        state = x * (1-alpha) + state * alpha
        x = (x - state) / 40
    with initial state linearly spaced from -60 to -90 dB.
    """
    x = np.asarray(x, dtype=np.float32)
    if NUMBA_AVAILABLE:
        return _erb_norm_numba(x, alpha)

    out = np.empty_like(x, dtype=np.float32)
    state = np.linspace(-60.0, -90.0, x.shape[-1], dtype=np.float32)
    for t in range(x.shape[0]):
        state = x[t] * (1.0 - alpha) + state * alpha
        out[t] = (x[t] - state) / 40.0
    return out


def unit_norm(spec: np.ndarray,
              alpha: float,
              eps: float = 1e-12) -> np.ndarray:
    """Normalize complex STFT bins with an EMA of the magnitude."""
    spec = np.asarray(spec, dtype=np.complex64)
    if NUMBA_AVAILABLE:
        return _unit_norm_numba(spec, alpha, eps)

    out = np.empty_like(spec, dtype=np.complex64)
    mag = np.abs(spec)
    state = np.linspace(0.001, 0.0001, spec.shape[-1], dtype=np.float32)
    for t in range(spec.shape[0]):
        state = mag[t] * (1.0 - alpha) + state * alpha
        out[t] = spec[t] / np.sqrt(np.maximum(state, eps))
    return out


def vorbis_window(n_fft: int) -> np.ndarray:
    """Return DeepFilterNet/libDF's Vorbis analysis/synthesis window."""
    half = n_fft // 2
    idx = np.arange(n_fft, dtype=np.float64)
    sin = np.sin(0.5 * np.pi * (idx + 0.5) / half)
    return np.sin(0.5 * np.pi * sin * sin).astype(np.float32)


def stft(wave: np.ndarray,
         n_fft: int,
         hop_length: int,
         win_length: int,
         pad: bool = True) -> np.ndarray:
    """Streaming STFT compatible with libDF's DFState.analysis.

    The output is shaped [F, T], matching librosa-style callers in this port,
    but the framing, Vorbis window, and FFT normalization follow libDF.
    """
    if win_length != n_fft:
        raise ValueError("DeepFilterNet STFT expects win_length == n_fft")

    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if pad:
        wave = np.pad(wave, (0, n_fft))

    window = vorbis_window(n_fft)
    wnorm = 1.0 / (float(n_fft * n_fft) / float(2 * hop_length))
    if NUMBA_AVAILABLE:
        frames = _analysis_frames_numba(wave, window, n_fft, hop_length, wnorm)
        return np.fft.rfft(frames, n=n_fft, axis=1).T.astype(np.complex64)

    n_frames = int(len(wave) // hop_length)
    mem_len = n_fft - hop_length
    analysis_mem = np.zeros(mem_len, dtype=np.float32)
    specs = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        frame = wave[start:start + hop_length]
        if frame.shape[0] < hop_length:
            frame = np.pad(frame, (0, hop_length - frame.shape[0]))

        buf = np.empty(n_fft, dtype=np.float32)
        buf[:mem_len] = analysis_mem * window[:mem_len]
        buf[mem_len:] = frame * window[mem_len:]

        analysis_split = mem_len - hop_length
        if analysis_split > 0:
            analysis_mem = np.roll(analysis_mem, -hop_length)
        analysis_mem[analysis_split:] = frame

        specs[:, frame_idx] = (np.fft.rfft(buf).astype(np.complex64) * wnorm)

    return specs


def istft(spec: np.ndarray,
          hop_length: int,
          win_length: int,
          length: Optional[int] = None) -> np.ndarray:
    """Streaming inverse STFT compatible with libDF's DFState.synthesis."""
    spec = np.asarray(spec, dtype=np.complex64)
    n_fft = (spec.shape[0] - 1) * 2
    if win_length != n_fft:
        raise ValueError("DeepFilterNet ISTFT expects win_length == n_fft")

    window = vorbis_window(n_fft)
    if NUMBA_AVAILABLE:
        frames_td = np.fft.irfft(spec.T, n=n_fft, axis=1).astype(np.float32)
        frames_td *= n_fft
        frames_td *= window[None, :]
        audio = _synthesis_overlap_add_numba(frames_td, hop_length)
        if length is not None:
            audio = audio[:length]
        return audio

    mem_len = n_fft - hop_length
    synthesis_mem = np.zeros(mem_len, dtype=np.float32)
    frames = []

    for frame_idx in range(spec.shape[1]):
        frame_td = np.fft.irfft(spec[:, frame_idx], n=n_fft).astype(np.float32)
        frame_td *= n_fft
        frame_td *= window

        out = frame_td[:hop_length] + synthesis_mem[:hop_length]
        frames.append(out)

        split = mem_len - hop_length
        x_second = frame_td[hop_length:]
        if split > 0:
            synthesis_mem = np.roll(synthesis_mem, -hop_length)
        synthesis_mem[:split] += x_second[:split]
        synthesis_mem[split:] = x_second[split:]

    audio = np.concatenate(frames).astype(np.float32)
    if length is not None:
        audio = audio[:length]
    return audio


def make_features(spec_ft: np.ndarray,
                  erb_widths: np.ndarray,
                  nb_df: int,
                  alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ONNX inputs from a complex STFT shaped [F, T]."""
    spec_tf = spec_ft.T.astype(np.complex64)
    erb_fb = erb_filterbank_from_widths(erb_widths, normalized=True, inverse=False)
    erb_feat = (np.abs(spec_tf) ** 2) @ erb_fb
    erb_feat = 10.0 * np.log10(erb_feat + 1e-10)
    erb_feat = erb_norm(erb_feat, alpha)

    spec_feat_c = unit_norm(spec_tf[:, :nb_df], alpha)
    spec_feat = np.stack((spec_feat_c.real, spec_feat_c.imag), axis=-1)
    spec_ri = np.stack((spec_tf.real, spec_tf.imag), axis=-1)

    return (
        spec_ri[None, None].astype(np.float32),
        erb_feat[None, None].astype(np.float32),
        spec_feat[None, None].astype(np.float32),
    )
