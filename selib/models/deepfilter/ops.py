"""NumPy post-processing operations for DeepFilterNet ONNX outputs.

When numba is installed, the deep-filter FIR loop is compiled automatically.
Vectorized NumPy remains the fallback for environments without numba.
"""

import numpy as np

from .features import erb_filterbank_from_widths

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba installed.
    njit = None

NUMBA_AVAILABLE = njit is not None


if NUMBA_AVAILABLE:
    @njit
    def _apply_deep_filter_numba(spec_tf: np.ndarray,
                                 coefs_c: np.ndarray,
                                 df_bins: int,
                                 df_order: int,
                                 df_lookahead: int,
                                 alpha: np.ndarray,
                                 use_alpha: bool) -> np.ndarray:
        out = spec_tf.copy()
        n_frames = spec_tf.shape[0]
        pad_left = df_order - df_lookahead - 1

        for frame in range(n_frames):
            for bin_idx in range(df_bins):
                filtered = np.complex64(0.0 + 0.0j)
                for order in range(df_order):
                    src_frame = frame + order - pad_left
                    if 0 <= src_frame < n_frames:
                        filtered += spec_tf[src_frame, bin_idx] * coefs_c[frame, order, bin_idx]
                if use_alpha:
                    a = alpha[frame, 0]
                    out[frame, bin_idx] = filtered * a + spec_tf[frame, bin_idx] * (1.0 - a)
                else:
                    out[frame, bin_idx] = filtered
        return out


def apply_erb_mask(spec_tf: np.ndarray,
                   erb_mask: np.ndarray,
                   erb_widths: np.ndarray,
                   post_filter: bool = False,
                   post_filter_beta: float = 0.02,
                   eps: float = 1e-12) -> np.ndarray:
    """Apply an ERB-band mask to a complex spectrogram.

    The model predicts one mask value per ERB band. This function expands that
    band mask back to FFT bins and multiplies it with the complex input
    spectrum, preserving phase.

    Parameters
    ----------
    spec_tf : ndarray
        Complex spectrogram shaped ``[frames, freq]``.
    erb_mask : ndarray
        ERB-band mask shaped ``[frames, bands]``.
    erb_widths : ndarray
        Number of FFT bins assigned to each ERB band.
    post_filter : bool, default=False
        If True, apply the optional DeepFilterNet post-filter curve to the
        expanded mask.
    post_filter_beta : float, default=0.02
        Strength parameter for the optional post-filter.
    eps : float, default=1e-12
        Numerical floor used by the post-filter.

    Returns
    -------
    ndarray
        Masked complex spectrogram shaped ``[frames, freq]``.
    """
    inv_fb = erb_filterbank_from_widths(erb_widths, normalized=True, inverse=True)
    mask = erb_mask @ inv_fb
    if post_filter:
        mask_sin = mask * np.sin(np.pi * mask / 2.0)
        mask = (1.0 + post_filter_beta) * mask / (
            1.0 + post_filter_beta * (mask / np.maximum(mask_sin, eps)) ** 2
        )
    return spec_tf * mask.astype(np.float32)


def apply_deep_filter(spec_tf: np.ndarray,
                      coefs: np.ndarray,
                      df_bins: int,
                      df_order: int,
                      df_lookahead: int = 0,
                      alpha: np.ndarray = None) -> np.ndarray:
    """Apply complex deep-filter coefficients.

    Deep filtering predicts a short complex FIR filter per time-frequency bin.
    The filter is applied only to the first ``df_bins`` low-frequency bins; the
    remaining bins are copied from ``spec_tf``.

    Parameters
    ----------
    spec_tf : ndarray
        Complex spectrogram shaped ``[frames, freq]``.
    coefs : ndarray
        Real/imaginary filter coefficients shaped ``[frames, order, df_bins, 2]``.
    df_bins : int
        Number of low-frequency bins to filter.
    df_order : int
        Number of complex filter taps per filtered bin.
    df_lookahead : int, default=0
        Number of future frames used by the filter alignment.
    alpha : ndarray, optional
        Optional blend factor shaped ``[frames, 1]``. If present, the filtered
        result is mixed with the original spectrum.

    Returns
    -------
    ndarray
        Complex spectrogram with the first ``df_bins`` bins deep-filtered.
    """
    spec_tf = np.asarray(spec_tf, dtype=np.complex64)
    coefs_c = coefs[..., 0].astype(np.float32) + 1j * coefs[..., 1].astype(np.float32)
    coefs_c = coefs_c.astype(np.complex64)
    if NUMBA_AVAILABLE:
        if alpha is None:
            alpha_numba = np.empty((spec_tf.shape[0], 1), dtype=np.float32)
            use_alpha = False
        else:
            alpha_numba = alpha.reshape(-1, 1).astype(np.float32)
            use_alpha = True
        return _apply_deep_filter_numba(
            spec_tf,
            coefs_c,
            df_bins,
            df_order,
            df_lookahead,
            alpha_numba,
            use_alpha,
        )

    pad_left = df_order - df_lookahead - 1
    pad_right = df_lookahead
    padded = np.pad(spec_tf[:, :df_bins], ((pad_left, pad_right), (0, 0)))

    filtered = np.zeros((spec_tf.shape[0], df_bins), dtype=np.complex64)
    for order in range(df_order):
        filtered += padded[order:order + spec_tf.shape[0]] * coefs_c[:, order]

    out = spec_tf.copy()
    if alpha is not None:
        a = alpha.reshape(-1, 1).astype(np.float32)
        out[:, :df_bins] = filtered * a + spec_tf[:, :df_bins] * (1.0 - a)
    else:
        out[:, :df_bins] = filtered
    return out


def attenuation_limit(noisy_tf: np.ndarray,
                      enhanced_tf: np.ndarray,
                      atten_lim_db: float = None) -> np.ndarray:
    """Limit how far the enhanced spectrum can move below the noisy spectrum.

    DeepFilterNet can strongly suppress time-frequency bins. That is usually
    desirable, but very deep attenuation may introduce musical noise or make the
    result sound over-processed. This helper applies the same style of
    attenuation limit used by the original enhancer by blending a fixed amount
    of the noisy complex spectrum back into the enhanced spectrum.

    Parameters
    ----------
    noisy_tf : ndarray
        Original noisy complex spectrogram shaped ``[frames, freq]``.
    enhanced_tf : ndarray
        Enhanced complex spectrogram with the same shape as ``noisy_tf``.
    atten_lim_db : float, optional
        Maximum attenuation in dB. ``atten_lim_db=12`` means the output keeps
        roughly a -12 dB copy of the noisy spectrum in every bin. ``None`` or a
        non-positive value disables the limit.

    Returns
    -------
    ndarray
        The enhanced spectrogram, optionally blended with the noisy spectrum.
    """
    if atten_lim_db is None or abs(atten_lim_db) <= 0:
        return enhanced_tf
    lim = 10 ** (-abs(atten_lim_db) / 20)
    return noisy_tf * lim + enhanced_tf * (1.0 - lim)
