"""Lightweight NumPy metrics for speech enhancement.

These metrics do not depend on external perceptual-metric packages such as
PESQ, STOI, or DNSMOS. They are useful for quick checks, regression tests, and
objective comparisons when a clean reference signal is available.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import warnings

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]
_WARNED_UNEQUAL_LENGTH = False


def _as_1d_float(name: str, x: ArrayLike) -> np.ndarray:
    """Convert input audio to a contiguous 1-D float64 array."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return x


def _align_pair(reference: ArrayLike,
                estimate: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert and truncate two signals to a shared length."""
    global _WARNED_UNEQUAL_LENGTH
    reference = _as_1d_float("reference", reference)
    estimate = _as_1d_float("estimate", estimate)
    n = min(reference.size, estimate.size)
    if n == 0:
        raise ValueError("reference and estimate must contain samples")
    if reference.size != estimate.size and not _WARNED_UNEQUAL_LENGTH:
        warnings.warn(
            "reference and estimate have unequal lengths "
            f"({reference.size} vs {estimate.size}); aligning by truncating "
            f"both signals to the first {n} samples.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_UNEQUAL_LENGTH = True
    return reference[:n], estimate[:n]


def power_db(x: ArrayLike, eps: float = 1e-12) -> float:
    """Return average signal power in dB.

    Parameters
    ----------
    x : ndarray
        Input waveform.
    eps : float, default=1e-12
        Numerical floor applied before ``log10``.

    Returns
    -------
    float
        ``10 * log10(mean(x ** 2))`` in dB.
    """
    x = _as_1d_float("x", x)
    return float(10.0 * np.log10(np.mean(np.square(x)) + eps))


def rms(x: ArrayLike) -> float:
    """Return root-mean-square amplitude."""
    x = _as_1d_float("x", x)
    return float(np.sqrt(np.mean(np.square(x))))


def mse(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Return mean squared error between two aligned waveforms."""
    reference, estimate = _align_pair(reference, estimate)
    return float(np.mean(np.square(reference - estimate)))


def rmse(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Return root mean squared error between two aligned waveforms."""
    return float(np.sqrt(mse(reference, estimate)))


def mae(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Return mean absolute error between two aligned waveforms."""
    reference, estimate = _align_pair(reference, estimate)
    return float(np.mean(np.abs(reference - estimate)))


def snr(reference: ArrayLike,
        estimate: ArrayLike,
        eps: float = 1e-12) -> float:
    """Return signal-to-noise ratio in dB.

    ``reference`` is normally the clean speech signal. ``estimate`` may be a
    noisy or enhanced signal. The residual ``reference - estimate`` is treated
    as noise/error.

    Parameters
    ----------
    reference : ndarray
        Clean reference waveform.
    estimate : ndarray
        Estimated waveform to compare against ``reference``.
    eps : float, default=1e-12
        Numerical floor for the denominator.

    Returns
    -------
    float
        SNR in dB.
    """
    reference, estimate = _align_pair(reference, estimate)
    signal_power = np.sum(np.square(reference))
    noise_power = np.sum(np.square(reference - estimate))
    return float(10.0 * np.log10((signal_power + eps) / (noise_power + eps)))


def segmental_snr(reference: ArrayLike,
                  estimate: ArrayLike,
                  sample_rate: int = 16000,
                  frame_ms: float = 20.0,
                  hop_ms: Optional[float] = None,
                  min_snr_db: float = -10.0,
                  max_snr_db: float = 35.0,
                  eps: float = 1e-12) -> float:
    """Return mean clipped frame-wise SNR in dB.

    Segmental SNR computes SNR per short frame and averages the clipped values.
    Clipping keeps silent or near-silent frames from dominating the score.

    Parameters
    ----------
    reference : ndarray
        Clean reference waveform.
    estimate : ndarray
        Estimated waveform to compare against ``reference``.
    sample_rate : int, default=16000
        Sample rate in Hz, used to convert frame and hop durations to samples.
    frame_ms : float, default=20.0
        Frame length in milliseconds.
    hop_ms : float, optional
        Hop length in milliseconds. Defaults to ``frame_ms``.
    min_snr_db : float, default=-10.0
        Lower clipping bound for each frame SNR.
    max_snr_db : float, default=35.0
        Upper clipping bound for each frame SNR.
    eps : float, default=1e-12
        Numerical floor for frame powers.

    Returns
    -------
    float
        Mean clipped segmental SNR in dB.
    """
    reference, estimate = _align_pair(reference, estimate)
    if hop_ms is None:
        hop_ms = frame_ms

    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if reference.size < frame_len:
        return snr(reference, estimate, eps=eps)

    values = []
    for start in range(0, reference.size - frame_len + 1, hop_len):
        ref_frame = reference[start:start + frame_len]
        err_frame = ref_frame - estimate[start:start + frame_len]
        frame_snr = 10.0 * np.log10(
            (np.sum(np.square(ref_frame)) + eps)
            / (np.sum(np.square(err_frame)) + eps)
        )
        values.append(np.clip(frame_snr, min_snr_db, max_snr_db))
    return float(np.mean(values))


def segsnr(reference: ArrayLike,
           estimate: ArrayLike,
           sample_rate: int = 16000,
           frame_ms: float = 20.0,
           hop_ms: Optional[float] = None,
           min_snr_db: float = -10.0,
           max_snr_db: float = 35.0,
           eps: float = 1e-12) -> float:
    """Alias for ``segmental_snr``."""
    return segmental_snr(
        reference,
        estimate,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        eps=eps,
    )


def si_sdr(reference: ArrayLike,
           estimate: ArrayLike,
           zero_mean: bool = True,
           eps: float = 1e-12) -> float:
    """Return scale-invariant signal-to-distortion ratio in dB.

    SI-SDR removes gain differences between ``reference`` and ``estimate``
    before measuring residual error. This is often more useful than plain SNR
    when an enhancement model changes output loudness.

    Parameters
    ----------
    reference : ndarray
        Clean reference waveform.
    estimate : ndarray
        Estimated waveform to compare against ``reference``.
    zero_mean : bool, default=True
        If True, remove the mean from both signals before projection.
    eps : float, default=1e-12
        Numerical floor for divisions and logarithms.

    Returns
    -------
    float
        SI-SDR in dB.
    """
    reference, estimate = _align_pair(reference, estimate)
    if zero_mean:
        reference = reference - np.mean(reference)
        estimate = estimate - np.mean(estimate)

    scale = np.sum(estimate * reference) / (np.sum(np.square(reference)) + eps)
    target = scale * reference
    noise = estimate - target
    return float(
        10.0 * np.log10(
            (np.sum(np.square(target)) + eps)
            / (np.sum(np.square(noise)) + eps)
        )
    )


def si_snr(reference: ArrayLike,
           estimate: ArrayLike,
           zero_mean: bool = True,
           eps: float = 1e-12) -> float:
    """Alias for ``si_sdr``."""
    return si_sdr(reference, estimate, zero_mean=zero_mean, eps=eps)
