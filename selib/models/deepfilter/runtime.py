"""Offline DeepFilterNet ONNX inference.

This module provides a small runtime for Option-B DeepFilterNet exports. The
ONNX model predicts ERB masks and deep-filter coefficients only; STFT analysis,
feature normalization, ERB masking, deep filtering, and synthesis are handled
with NumPy in this package.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union, Any, Literal

import numpy as np
import onnxruntime as ort

from ..core import get_onnx_providers, resolve_model_path, warn_if_cuda_inactive
from .features import istft, make_features, stft
from .ops import apply_deep_filter, apply_erb_mask, attenuation_limit

DF_MODEL_TYPE = Literal["deepfilternet1", "deepfilternet2", "deepfilternet3",]


class DeepFilterNetOnnx:
    """Run a DeepFilterNet Option-B ONNX model on mono waveforms.

    The ONNX file must contain ``selib_config`` metadata, as written by the
    exporter in this repository. The metadata supplies the sample rate, FFT and
    hop sizes, ERB band layout, deep-filter order, and model-family switches
    needed to reproduce the original DeepFilterNet post-processing path.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Existing local ONNX path or model id from ``selib.urls.MODEL_URLS``. If
        the local path does not exist and the value is a known model id, the
        ONNX file is downloaded and cached before loading.
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
                 model_path: Union[str, Path, DF_MODEL_TYPE],
                 providers: Optional[list] = None,
                 cuda: bool = False,
                 cache_dir: Optional[Union[str, Path]] = None,
                 verbose: bool = True) -> None:
        self.model_path = resolve_model_path(
            model_path, cache_dir=cache_dir, verbose=verbose)
        providers = get_onnx_providers(providers=providers, cuda=cuda)
        self.sess = ort.InferenceSession(
            str(self.model_path), providers=providers)
        warn_if_cuda_inactive(self.sess, cuda=cuda)
        self.input_names = [x.name for x in self.sess.get_inputs()]
        self.output_names = [x.name for x in self.sess.get_outputs()]

        meta = self.sess.get_modelmeta()
        self.config: Dict[str, object] = json.loads(
            meta.custom_metadata_map.get("selib_config", "{}"))
        if not self.config:
            raise ValueError(
                f"No selib_config metadata found in {self.model_path}")

        self.sr = int(self.config["sr"])
        self.fft_size = int(self.config["fft_size"])
        self.hop_size = int(self.config["hop_size"])
        self.win_size = int(self.config.get("win_size", self.fft_size))
        self.nb_df = int(self.config["nb_df"])
        self.df_order = int(self.config["df_order"])
        self.df_lookahead = int(self.config["df_lookahead"])
        self.df_iter = int(self.config.get("df_iter", 1))
        self.norm_alpha = float(self.config["norm_alpha"])
        self.erb_widths = np.asarray(self.config["erb_widths"], dtype=np.int64)
        self.model_family = str(self.config.get("model_family", "unknown"))
        self.use_alpha = bool(self.config.get("use_alpha", False))
        self.run_erb = bool(self.config.get("run_erb", True))
        self.run_df = bool(self.config.get("run_df", True))
        self.post_filter = bool(self.config.get("post_filter", False))
        self.post_filter_beta = float(
            self.config.get("post_filter_beta", 0.02))

    def forward_features(self,
                         spec_ri: np.ndarray,
                         erb_feat: np.ndarray,
                         spec_feat: np.ndarray) -> Dict[str, np.ndarray]:
        """Run the ONNX network on precomputed DeepFilterNet features.

        Inputs are matched to ONNX graph inputs by name. This matters because
        DeepFilterNet1/2 exports may not keep the full complex spectrogram input
        when the graph does not use it.

        Parameters
        ----------
        spec_ri : ndarray
            Real/imaginary noisy spectrogram shaped ``[1, 1, frames, freq, 2]``.
        erb_feat : ndarray
            Normalized ERB feature tensor shaped ``[1, 1, frames, bands]``.
        spec_feat : ndarray
            Normalized low-frequency complex feature tensor shaped
            ``[1, 1, frames, nb_df, 2]``.

        Returns
        -------
        dict
            ONNX outputs keyed by output name, typically ``erb_mask``,
            ``df_coefs``, ``df_alpha``, and ``lsnr``.
        """
        input_values = {
            "spec": spec_ri,
            "erb_feat": erb_feat,
            "feat_erb": erb_feat,
            "spec_feat": spec_feat,
            "feat_spec": spec_feat,
        }
        feed = {}
        for name in self.input_names:
            if name not in input_values:
                raise ValueError(
                    f"Unsupported ONNX input {name!r}; expected one of "
                    f"{sorted(input_values)}")
            feed[name] = input_values[name].astype(np.float32, copy=False)

        outputs = self.sess.run(
            self.output_names,
            feed,
        )
        return dict(zip(self.output_names, outputs))

    def enhance(self,
                wave: np.ndarray,
                pad: bool = True,
                atten_lim_db: Optional[float] = None,
                return_dict: bool = False
                ) -> Union[np.ndarray, dict[str]]:
        """Enhance a mono waveform with the loaded DeepFilterNet model.

        The waveform is analyzed with the libDF-compatible STFT in
        ``features.py``, passed through the ONNX model, enhanced with the ERB
        mask and deep-filter coefficients, and synthesized back to audio.

        Parameters
        ----------
        wave : ndarray
            Input waveform at the model sample rate. For multidimensional input
            shaped ``[..., samples]``, the leading dimensions are flattened into
            the ONNX batch axis and restored in the output.
        pad : bool, default=True
            If True, append one FFT window before analysis and crop the
            algorithmic delay after synthesis, matching the original offline
            DeepFilterNet enhancement path.
        atten_lim_db : float, optional
            Maximum attenuation in dB, implemented as a final blend between the
            noisy and enhanced complex spectra. For example, ``atten_lim_db=12``
            keeps at least about -12 dB of the original noisy spectrum in every
            bin, which can reduce musical-noise artifacts but leaves more
            residual noise. ``None`` disables this limiting and uses the model
            output directly.
        return_dict : bool, default=False
            If True, return audio plus intermediate spectrograms and raw ONNX
            outputs for inspection.

        Returns
        -------
        ndarray | dict
            Enhanced waveform by default. With ``return_dict=True``, returns a
            dictionary containing ``audio``, noisy/enhanced spectrograms, ERB
            mask, deep-filter coefficients, optional alpha, and ONNX outputs.
        """
        # Accept mono, channel, or batched audio as (..., samples). The DSP
        # state is per waveform, while ONNX inference can use one batch axis.
        wave = np.asarray(wave, dtype=np.float32)
        input_shape = wave.shape
        if wave.ndim == 0:
            raise ValueError("wave must contain at least one sample")

        flat_waves = wave.reshape(-1, input_shape[-1])
        orig_len = input_shape[-1]

        # Analysis frontend: convert each waveform to the libDF-compatible
        # complex STFT and derive the ERB/spec features expected by the ONNX graph.
        spec_fts = [
            stft(
                flat_wave,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_size,
                pad=pad,
            )
            for flat_wave in flat_waves
        ]
        spec_tfs = [spec_ft.T.astype(np.complex64) for spec_ft in spec_fts]
        features = [
            make_features(spec_ft, self.erb_widths, self.nb_df, self.norm_alpha)
            for spec_ft in spec_fts
        ]
        spec_ri = np.concatenate([feat[0] for feat in features], axis=0)
        erb_feat = np.concatenate([feat[1] for feat in features], axis=0)
        spec_feat = np.concatenate([feat[2] for feat in features], axis=0)

        # Network stage: run the exported model once with the flattened batch.
        # Outputs are ERB masks and complex deep-filter coefficients.
        out = self.forward_features(spec_ri, erb_feat, spec_feat)
        masks = out["erb_mask"][:, 0]
        coefs_batch = out["df_coefs"]
        alpha_batch = out.get("df_alpha", None)

        audios = []
        specs_enh = []
        alphas = []
        for batch_idx, spec_tf in enumerate(spec_tfs):
            # Post-processing remains per waveform because ERB masking, deep
            # filtering, and synthesis all operate on one complex spectrogram.
            mask = masks[batch_idx]
            coefs = coefs_batch[batch_idx]
            alpha = None
            if alpha_batch is not None and alpha_batch.size != 1:
                alpha = alpha_batch[batch_idx]
            alphas.append(alpha)

            if self.run_erb:
                # Expand the low-resolution ERB-band mask back to FFT bins.
                spec_masked = apply_erb_mask(
                    spec_tf,
                    mask,
                    self.erb_widths,
                    post_filter=False,
                    post_filter_beta=self.post_filter_beta,
                )
            else:
                spec_masked = spec_tf.copy()

            if self.run_df:
                # Apply the learned complex FIR filters to the low-frequency
                # bins. DF3 filters the noisy spectrum, older variants filter
                # the ERB-masked spectrum.
                df_input = spec_tf if self.model_family == "DeepFilterNet3" else spec_masked
                spec_enh = df_input
                for _ in range(self.df_iter):
                    spec_enh = apply_deep_filter(
                        spec_enh,
                        coefs,
                        df_bins=self.nb_df,
                        df_order=self.df_order,
                        df_lookahead=self.df_lookahead,
                        alpha=alpha if self.use_alpha else None,
                    )
                if self.model_family == "DeepFilterNet3" and self.run_erb:
                    # DF3 uses deep filtering for low bins and the ERB mask for
                    # the remaining high-frequency bins.
                    spec_enh[:, self.nb_df:] = spec_masked[:, self.nb_df:]
            else:
                spec_enh = spec_masked

            if self.post_filter:
                # Optional DeepFilterNet post-filter: slightly strengthens
                # suppression where the inferred mask is already small.
                eps = 1e-12
                mag_ratio = np.abs(spec_enh) / np.maximum(np.abs(spec_tf), eps)
                mask_pf = np.clip(mag_ratio, eps, 1.0)
                mask_sin = mask_pf * np.sin(np.pi * mask_pf / 2.0)
                pf = (1.0 + self.post_filter_beta) / (
                    1.0 + self.post_filter_beta * (mask_pf / np.maximum(mask_sin, eps)) ** 2)
                spec_enh = spec_enh * pf

            # Optional safety blend with the noisy spectrum to avoid extremely
            # deep attenuation artifacts, then synthesize back to waveform.
            spec_enh = attenuation_limit(
                spec_tf, spec_enh, atten_lim_db=atten_lim_db)
            audio = istft(
                spec_enh.T,
                hop_length=self.hop_size,
                win_length=self.win_size,
                length=None,
            )
            if pad:
                delay = self.fft_size - self.hop_size
                audio = audio[delay:orig_len + delay]
            else:
                audio = audio[:orig_len]
            audios.append(audio.astype(np.float32))
            specs_enh.append(spec_enh)

        # Restore the original leading dimensions so (..., samples) in produces
        # (..., samples) out. Return dict entries follow the same leading dims.
        audio = np.stack(audios, axis=0)
        if wave.ndim == 1:
            audio_out = audio[0]
            spec_noisy_out = spec_tfs[0]
            spec_enh_out = specs_enh[0]
            mask_out = masks[0]
            coefs_out = coefs_batch[0]
            alpha_out = alphas[0]
        else:
            audio_out = audio.reshape(*input_shape[:-1], audio.shape[-1])
            spec_noisy_out = np.stack(spec_tfs, axis=0).reshape(
                *input_shape[:-1], *spec_tfs[0].shape)
            spec_enh_out = np.stack(specs_enh, axis=0).reshape(
                *input_shape[:-1], *specs_enh[0].shape)
            mask_out = masks.reshape(*input_shape[:-1], *masks.shape[1:])
            coefs_out = coefs_batch.reshape(*input_shape[:-1], *coefs_batch.shape[1:])
            if alpha_batch is None or alpha_batch.size == 1:
                alpha_out = None
            else:
                alpha_out = alpha_batch.reshape(*input_shape[:-1], *alpha_batch.shape[1:])

        if return_dict:
            return {
                "audio": audio_out.astype(np.float32),
                "spec_noisy": spec_noisy_out,
                "spec_enhanced": spec_enh_out,
                "erb_mask": mask_out,
                "df_coefs": coefs_out,
                "df_alpha": alpha_out,
                "onnx_outputs": out,
            }
        return audio_out.astype(np.float32)

    def __repr__(self) -> str:
        """Return the model metadata as a readable multi-line string."""
        return '\n'.join(f"{k}: {v}" for k, v in self.config.items())
