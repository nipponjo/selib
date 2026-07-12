"""ONNX speech-enhancement model wrappers.

This module contains small NumPy/librosa/ONNX Runtime wrappers for packaged
speech-enhancement models. The classes operate on mono waveforms and keep the
model-specific spectrogram details behind a simple ``enhance()`` method.
"""

import json
import numpy as np
import librosa
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Literal

from .core import get_onnx_providers, resolve_model_path, warn_if_cuda_inactive

MM_MODEL_ID = Literal[
    "ul_unas_16k",
]

class MagnitudeMaskModel:
    """Run an ONNX magnitude-mask speech-enhancement model.

    The model receives either a magnitude spectrogram or a real/imaginary
    spectrogram, predicts a multiplicative mask, and reuses the noisy phase for
    waveform reconstruction. Runtime parameters such as FFT size, hop length,
    sample rate, input type, and mask clamp range are read from the ONNX
    ``selib_config`` metadata when present. Missing metadata falls back to the
    legacy defaults.

    Parameters
    ----------
    model_path : str or pathlib.Path, default='./unet_d0.onnx'
        Existing local ONNX path or model id from ``selib.urls.MODEL_URLS``. If
        the local path does not exist and the value is a known model id, the
        ONNX file is downloaded and cached before loading. The model is expected
        to expose ``spec_abs`` for magnitude input or ``spec_ri`` for
        real/imaginary input.
    providers : list, optional
        Optional ONNX Runtime execution providers. Defaults to CPU execution.
        The list of providers is ordered by precedence. For example
        `['CUDAExecutionProvider', 'CPUExecutionProvider']`
        means execute a node using `CUDAExecutionProvider`
        if capable, otherwise execute using `CPUExecutionProvider`.
    cuda : bool, default=False
        If True and ``providers`` is not given, request
        ``['CUDAExecutionProvider', 'CPUExecutionProvider']``. A warning is
        printed if CUDA is requested but not active after session creation.
    cache_dir : str or pathlib.Path, optional
        Directory for downloaded models. Defaults to ``~/.cache/selib/models``.
    verbose : bool, default=True
        If True, print model download source, destination, and progress.
    """

    def __init__(self,
                 model_path: Union[str, Path, MM_MODEL_ID] = './unet_d0.onnx',
                 providers: Optional[list] = None,
                 cuda: bool = False,
                 cache_dir: Optional[Union[str, Path]] = None,
                 verbose: bool = True,
                 path: Optional[Union[str, Path]] = None,
                 ) -> None:
        if path is not None:
            model_path = path
        self.model_path = resolve_model_path(
            model_path, cache_dir=cache_dir, verbose=verbose)
        providers = get_onnx_providers(providers=providers, cuda=cuda)
        self.sess = ort.InferenceSession(
            str(self.model_path), providers=providers,)
        warn_if_cuda_inactive(self.sess, cuda=cuda)
        self.input_names = [x.name for x in self.sess.get_inputs()]
        self.output_names = [x.name for x in self.sess.get_outputs()]

        config = {}
        try:
            meta = self.sess.get_modelmeta()
            config = json.loads(meta.custom_metadata_map["selib_config"])
        except (KeyError, json.JSONDecodeError):
            print(
                f"warning: no model metadata found in model @ {model_path}, using default parameters")

        self.input_type = config.get("input", "magnitude_spectrogram")
        self.input_key = {
            'magnitude_spectrogram': 'spec_abs',
            'real_imag_spectrogram': 'spec_ri'
        }.get(self.input_type, 'spec_abs')

        # STFT arguments
        self.n_fft = int(config.get("n_fft", 512))
        self.n_hop = int(config.get("hop_length", 256))
        self.sr = int(config.get("sr", config.get("sample_rate", 22050)))
        # Mask clamping
        self.mask_clamp = config.get("mask_clamp", (0, 10))

        self.config = config

    def prepare_model_input(self, spec_input):
        """Convert a complex STFT into the ONNX input tensor.

        Parameters
        ----------
        spec_input : ndarray
            Complex STFT with shape ``[freq, frames]``.

        Returns
        -------
        ndarray
            Model input shaped either ``[1, 1, freq, frames]`` for magnitude
            input or ``[1, 2, freq, frames]`` for real/imaginary input.
        """
        if self.input_type == "magnitude_spectrogram":
            spec_input_abs = np.abs(spec_input)
            # spec_input_phase = np.angle(spec_input)
            model_input = spec_input_abs[None, None]
        elif self.input_type == "real_imag_spectrogram":
            spec_input_ri = np.stack(
                [spec_input.real, spec_input.imag], axis=0)
            model_input = spec_input_ri[None]
        return model_input

    def _prepare_batch_model_input(self, specs_input):
        """Convert batched STFTs into a flattened ONNX batch tensor."""
        freq_bins, frames = specs_input.shape[-2:]
        flat_specs = specs_input.reshape(-1, freq_bins, frames)
        if self.input_type == "magnitude_spectrogram":
            model_input = np.abs(flat_specs)[:, None]
        elif self.input_type == "real_imag_spectrogram":
            model_input = np.stack(
                [flat_specs.real, flat_specs.imag], axis=1)
        else:
            raise ValueError(f"Unsupported input_type {self.input_type!r}")
        return model_input

    def enhance(self,
                wave: np.ndarray,
                return_mask: bool = False,
                return_dict: bool = False,
                padding: Optional[bool] = True,
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
        """Enhance a waveform by applying the predicted magnitude mask.

        Parameters
        ----------
        wave : ndarray
            Waveform at the sample rate expected by the ONNX model. For
            multidimensional input shaped ``[..., samples]``, the leading
            dimensions are flattened into the ONNX batch axis and restored in
            the output.

        return_mask : bool, default=False
            If True, return both the enhanced waveform and the magnitude mask
            as ``(wave_enhanced, mask)``.

        return_dict : bool, default=False
            If True, return intermediate arrays useful for debugging or
            visualization. This takes precedence over ``return_mask``.

        padding : bool or None, default=True
            If not ``None``, force the reconstructed waveform to have the same
            number of samples as ``wave`` by passing ``length=len(wave)`` to
            ``librosa.istft()``. Use ``None`` to keep librosa's raw output
            length behavior.

        Returns
        -------
        ndarray | tuple | dict
            Enhanced waveform by default; ``(wave_enhanced, mask)`` when
            ``return_mask=True``; or a dictionary containing the waveform,
            mask, noisy STFT, and enhanced STFT when ``return_dict=True``.
        """
        wave = np.asarray(wave, dtype=np.float32)
        input_shape = wave.shape
        if wave.ndim == 0:
            raise ValueError("wave must contain at least one sample")

        output_length = input_shape[-1] if padding is not None else None

        # STFT
        specs_input = librosa.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            pad_mode='reflect'
        )

        model_input = self._prepare_batch_model_input(specs_input)

        # infer mask
        mask = self.sess.run(
            None, {self.input_key: model_input.astype(np.float32)})[0]
        mask = np.clip(mask, *self.mask_clamp)[:, 0]  # clamp mask

        # multiply
        freq_bins, frames = specs_input.shape[-2:]
        specs_input_flat = specs_input.reshape(-1, freq_bins, frames)
        specs_enhan_flat = specs_input_flat * mask
        specs_enhan = specs_enhan_flat.reshape(*specs_input.shape)

        # reuse phase
        # spec_enhan = spec_abs_enhan*np.exp(1j*spec_input_phase)

        # re-synthesize
        wave_enhan = librosa.istft(
            specs_enhan,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            length=output_length,
        )
        if wave.ndim == 1:
            mask_out = mask[0]
            spec_input_out = specs_input
            spec_enhan_out = specs_enhan
        else:
            mask_out = mask.reshape(*input_shape[:-1], *mask.shape[1:])
            spec_input_out = specs_input
            spec_enhan_out = specs_enhan
        if return_dict:
            return {
                'wave_enhanced': wave_enhan,
                'mask': mask_out,
                "spec_input": spec_input_out,
                'spec_enhanced': spec_enhan_out
            }
        if return_mask:
            return wave_enhan, mask_out
        return wave_enhan

    def __repr__(self) -> str:
        return '\n'.join(f"{k}: {v}" for k, v in self.config.items())
