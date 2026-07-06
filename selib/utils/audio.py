"""Small NumPy-based audio utilities.

The helpers in this module intentionally avoid heavy audio I/O dependencies.
They cover simple WAV writing, level normalization, SNR-based mixing, and a
mask-derived SNR estimate useful for speech-enhancement experiments.
"""

import wave
import numpy as np
from typing import Union, Optional

EPS = np.finfo(float).eps

def save_wave(audio_data: np.ndarray,
              audio_path: str,
              sample_rate: int = 22050,  # Sample rate in Hz
              bits_per_sample: int = 32,  # You can choose 16, 24 or 32 bits per sample
              normalize: bool = False
              ) -> None:
    """Save a mono waveform as an integer PCM WAV file.

    Parameters
    ----------
    audio_data : ndarray
        Mono waveform shaped ``[n_samples]``. Values are expected to be in
        approximately ``[-1, 1]`` unless ``normalize=True`` is used.
    audio_path : str
        Output WAV path.
    sample_rate : int, default=22050
        Sample rate in Hz.
    bits_per_sample : int, default=32
        Integer PCM bit depth. Supported values are 8, 16, 24, and 32.
    normalize : bool, default=False
        If True, scale the waveform by its absolute peak before quantization.
        The waveform is also peak-normalized automatically when its absolute
        peak is greater than 1.

    Notes
    -----
    24-bit audio is written as packed 3-byte little-endian PCM. Some media
    players display this as 32 bits/sample internally even though the file's
    sample width is 3 bytes.
    """
    assert bits_per_sample in (8, 16, 24, 32)
    assert audio_data.ndim == 1
      
    # Define the parameters of the audio file
    channels = 1  # Mono audio

    sample_width = bits_per_sample // 8
    int_value_max = (256**sample_width) // 2 - 1

    audio_abs_max = np.abs(audio_data).max()
    if audio_abs_max > 1 or normalize:
        audio_data /= audio_abs_max

    # Normalize the audio data to fit within the range of the chosen bit depth
    audio_data = audio_data * int_value_max

    if bits_per_sample == 24:
        audio_data = audio_data.astype(np.int32)
        audio_bytes = audio_data.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        audio_type = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        audio_bytes = audio_data.astype(audio_type).tobytes()

    # Create a new wave file
    with wave.open(audio_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

def active_rms(audio: np.ndarray, 
               window_size: int = 1024, 
               thresh: float = 0.0001
               ) -> float:
    """Estimate RMS level over active frames only.

    The waveform is split into non-overlapping frames. Frames whose mean square
    energy is greater than ``thresh`` are treated as active, and RMS is computed
    from those frames only. If no frame is active, ``EPS`` is returned.

    Parameters
    ----------
    audio : ndarray
        Mono waveform shaped ``[n_samples]``.
    window_size : int, default=1024
        Number of samples per analysis frame.
    thresh : float, default=0.0001
        Mean-square energy threshold used to select active frames.

    Returns
    -------
    float
        Active-frame RMS amplitude.
    """
    len_trunc = audio.shape[0] // window_size * window_size
    e_per_frame = (audio[:len_trunc].reshape(-1, window_size)**2).mean(1)
    e_gr_thresh = e_per_frame > thresh
    n_active_frames = e_gr_thresh.sum()  

    if n_active_frames == 0:
        return EPS
    active_rms = np.sqrt(e_per_frame[e_gr_thresh].sum() / n_active_frames)
    return active_rms    

def normalize(audio: np.ndarray, 
              target_level: float = -25
              ) -> np.ndarray:
    """Peak-normalize audio and scale it to a target active RMS level.

    Parameters
    ----------
    audio : ndarray
        Mono waveform shaped ``[n_samples]``.
    target_level : float, default=-25
        Target active RMS level in dBFS.

    Returns
    -------
    ndarray
        Normalized waveform.

    Notes
    -----
    This function modifies ``audio`` in place during peak normalization before
    returning the scaled result.
    """
    audio /= (np.abs(audio).max() + EPS)
    rms = active_rms(audio)
    scaler = 10**(target_level / 20) / (rms + EPS)
    return scaler * audio

def snr_mixer(clean: np.ndarray, 
              noise: np.ndarray,
              snr: float = 20
              ) -> np.ndarray:
    """Mix clean speech and noise at a requested SNR.

    Both inputs are first normalized with ``normalize()``. The noise is then
    scaled so that the mixture has approximately the requested SNR.

    Parameters
    ----------
    clean : ndarray
        Clean speech waveform.
    noise : ndarray
        Noise waveform. It should be at least as long as ``clean`` or already
        trimmed to the desired length.
    snr : float, default=20
        Target signal-to-noise ratio in dB.

    Returns
    -------
    ndarray
        Noisy speech mixture.
    """
    clean = normalize(clean)
    noise = normalize(noise)
 
    noise_newlevel = noise / (10**(snr/20))

    noisyspeech = clean + noise_newlevel
    return noisyspeech

def estimate_snr_from_mask_irm(
        spec_noisy_mag: np.ndarray, 
        mask: np.ndarray, 
        eps: float = 1e-8,
        axis = None
        ) -> Union[float, np.ndarray]:
    """Estimate SNR from a magnitude spectrogram and ideal-ratio-style mask.

    The mask is clipped to ``[0, 1]`` and interpreted as the speech magnitude
    proportion. Speech power is estimated as ``spec_noisy_mag**2 * mask**2``;
    noise power is estimated as ``spec_noisy_mag**2 * (1 - mask**2)``.

    Parameters
    ----------
    spec_noisy_mag : ndarray
        Noisy magnitude spectrogram.
    mask : ndarray
        Speech mask broadcastable to ``spec_noisy_mag``.
    eps : float, default=1e-8
        Numerical floor for the estimated noise power.
    axis : int or tuple of int, optional
        Axis or axes over which to sum power. ``None`` returns one global SNR.

    Returns
    -------
    float or ndarray
        Estimated SNR in dB.
    """
    mask = mask.clip(0, 1)

    speech_power = (spec_noisy_mag**2 * mask**2).sum(axis=axis)
    noise_power = (spec_noisy_mag**2 * (1 - mask**2)).sum(axis=axis)

    return 10 * np.log10(speech_power / noise_power.clip(eps))
