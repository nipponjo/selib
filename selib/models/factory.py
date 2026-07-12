"""Factory helpers for loading and running registered enhancement models."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np

from ..urls import MODEL_URLS
from ..utils.audio import save_wave
from .deepfilter import DeepFilterNetOnnx
from .se_models import MagnitudeMaskModel

MODEL_KIND_TO_CLASS = {
    "deepfilter": DeepFilterNetOnnx,
    "magnitude_mask": MagnitudeMaskModel,
}

SE_MODEL_ID = Literal[
    "deepfilternet1",
    "deepfilternet2",
    "deepfilternet3",
    "ul_unas_16k",
]

def get_model_kind(model_id: SE_MODEL_ID) -> str:
    """Return the registered model kind for a model id.

    Parameters
    ----------
    model_id : str
        Key from ``selib.urls.MODEL_URLS``.

    Returns
    -------
    str
        Model kind, for example ``"deepfilter"`` or ``"magnitude_mask"``.
    """
    if model_id not in MODEL_URLS:
        keys = ", ".join(sorted(MODEL_URLS))
        raise KeyError(f"Unknown model id {model_id!r}. Known model ids: {keys}")

    kind = str(MODEL_URLS[model_id].get("kind", ""))
    if not kind:
        raise ValueError(f"Model id {model_id!r} has no 'kind' in selib.urls.MODEL_URLS")
    if kind not in MODEL_KIND_TO_CLASS:
        kinds = ", ".join(sorted(MODEL_KIND_TO_CLASS))
        raise ValueError(
            f"Model id {model_id!r} has unsupported kind {kind!r}. "
            f"Supported kinds: {kinds}")
    return kind


def load_model(model_id: SE_MODEL_ID, **model_kwargs: Any):
    """Instantiate the wrapper class registered for ``model_id``.

    Parameters
    ----------
    model_id : str
        Key from ``selib.urls.MODEL_URLS``.
    **model_kwargs
        Extra keyword arguments passed to the selected wrapper constructor, such
        as ``cache_dir`` or ONNX Runtime ``providers``.

    Returns
    -------
    object
        A model wrapper with an ``enhance()`` method.
    """
    kind = get_model_kind(model_id)
    model_cls = MODEL_KIND_TO_CLASS[kind]
    return model_cls(model_id, **model_kwargs)


def _model_cache_key(model_id: SE_MODEL_ID, model_kwargs: Optional[dict[str, Any]]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Create a stable-enough cache key for model constructor arguments."""
    kwargs_items = tuple(
        sorted((key, repr(value)) for key, value in (model_kwargs or {}).items())
    )
    return model_id, kwargs_items


def _get_cached_model(model_id: SE_MODEL_ID, model_kwargs: Optional[dict[str, Any]]):
    """Load a model once and reuse it on later calls with the same arguments."""
    cache_key = _model_cache_key(model_id, model_kwargs)
    cache = getattr(enhance, "models", {})
    if cache_key not in cache:
        cache[cache_key] = load_model(model_id, **(model_kwargs or {}))
        setattr(enhance, "models", cache)
    return cache[cache_key]


def _wave_from_output(output: Union[np.ndarray, tuple, dict]) -> np.ndarray:
    """Extract the waveform from any supported model ``enhance()`` output."""
    if isinstance(output, dict):
        if "audio" in output:
            return output["audio"]
        if "wave_enhanced" in output:
            return output["wave_enhanced"]
        raise KeyError("Could not find 'audio' or 'wave_enhanced' in enhance output dict")
    if isinstance(output, tuple):
        return output[0]
    return output


def enhance(wave: np.ndarray,
            model_id: SE_MODEL_ID,
            model: Optional[object] = None,
            model_kwargs: Optional[dict[str, Any]] = None,
            cache_model: bool = True,
            cuda: bool = False,
            save_to: Optional[str] = None,
            bits_per_sample: Literal[8, 16, 24, 32] = 32,
            **enhance_kwargs: Any
            ) -> Union[np.ndarray, dict]:
    """Enhance a waveform with a registered model id.

    This is the shortest path for one-off inference. By default, the selected
    model wrapper is lazily loaded and cached per ``model_id`` plus
    ``model_kwargs`` so repeated calls do not reload the ONNX session.

    Parameters
    ----------
    wave : ndarray
        Mono waveform at the sample rate expected by the selected model.
    model_id : str
        Key from ``selib.urls.MODEL_URLS``.
    model : object, optional
        Already-loaded model wrapper. If omitted, the wrapper is constructed
        from ``model_id``.
    model_kwargs : dict, optional
        Constructor keyword arguments used only when ``model`` is omitted, such
        as ``cache_dir`` or ONNX Runtime ``providers``.
    cache_model : bool, default=True
        If True, cache the loaded model wrapper on the function and reuse it on
        later calls with the same ``model_id`` and ``model_kwargs``.
    cuda : bool, default=False
        If True and ``model`` is omitted, request CUDA execution for the loaded
        ONNX model. This sets the constructor default providers to
        ``['CUDAExecutionProvider', 'CPUExecutionProvider']`` unless providers
        are explicitly supplied in ``model_kwargs``.
    save_to : Optional[str], default=None
        If provided, save the enhanced waveform as a WAV file at this path.
    bits_per_sample : int, default=32
        Bit depth of the saved WAV file when ``save_to`` is specified.
        Supported values: 8, 16, 24 or 32.
    **enhance_kwargs
        Keyword arguments passed to the model wrapper's ``enhance()`` method,
        such as ``return_dict`` or ``atten_lim_db``.

    Returns
    -------
    ndarray | dict
        The selected model's enhancement result.
    """
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("cuda", cuda)
    if model is None:
        if cache_model:
            model = _get_cached_model(model_id, model_kwargs)
        else:
            model = load_model(model_id, **model_kwargs)

    output = model.enhance(wave, **enhance_kwargs)
    if save_to is not None:
        wave_out = _wave_from_output(output)
        if np.asarray(wave_out).ndim != 1:
            raise ValueError(
                "save_to only supports a single 1-D waveform. "
                "For batched/channel input, save each output item separately.")
        save_wave(
            np.asarray(wave_out, dtype=np.float32),
            str(save_to),
            sample_rate=int(getattr(model, "sr", 22050)),
            bits_per_sample=bits_per_sample,
        )
    return output
