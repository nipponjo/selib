import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

# reference:
# https://github.com/microsoft/DNS-Challenge/tree/2db96d5f75257df764a6ef66513b4b97bc707f30/DNSMOS


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
INPUT_SAMPLES = int(INPUT_LENGTH * SAMPLING_RATE)
ArrayLike = Union[np.ndarray, Sequence[float]]


def get_polyfit_val(sig: float,
                    bak: float,
                    ovr: float,
                    is_personalized_MOS: bool = False
                    ) -> Tuple[float, float, float]:
    """Apply the DNSMOS polynomial calibration to raw SIG/BAK/OVRL scores."""
    if is_personalized_MOS:
        p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
        p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
        p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
    else:
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

    return p_sig(sig), p_bak(bak), p_ovr(ovr)


class DnsMosSigBakOvrOnnx:
    """ONNX Runtime DNSMOS SIG/BAK/OVRL estimator.

    This wrapper mirrors the PyTorch DnsMosSigBakOvr forward(), infer(), and
    script() methods using only NumPy plus ONNX Runtime.
    """

    def __init__(self,
                 path: Optional[Union[str, Path]] = None,
                 providers: Optional[Iterable[str]] = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parents[1]
            path = path / "data" / "dnsmos" / "dnsmos_sig_bak_ovr2.onnx"

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.path: Path = Path(path)
        self.sess: ort.InferenceSession = ort.InferenceSession(
            str(self.path), providers=list(providers))
        self.input_name: str = self.sess.get_inputs()[0].name
        self.output_name: str = self.sess.get_outputs()[0].name

        self.config: Dict[str, object] = {}
        try:
            meta = self.sess.get_modelmeta()
            self.config = json.loads(
                meta.custom_metadata_map.get("dnsmos_config", "{}"))
        except Exception:
            self.config = {}

        self.sample_rate: int = int(
            self.config.get("sample_rate", SAMPLING_RATE))
        self.input_length: float = float(
            self.config.get("input_length", INPUT_LENGTH))
        self.input_samples: int = int(
            self.config.get("input_samples", INPUT_SAMPLES))

    def forward(self, waves: ArrayLike) -> np.ndarray:
        """Return raw DNSMOS scores for waveforms shaped [samples] or [B, samples]."""
        waves = np.asarray(waves, dtype=np.float32)
        if waves.ndim == 1:
            waves = waves[None, :]
        if waves.ndim != 2:
            raise ValueError("waves must have shape [samples] or [B, samples]")

        return self.sess.run(
            [self.output_name],
            {self.input_name: waves.astype(np.float32, copy=False)}
        )[0]

    def __call__(self, waves: ArrayLike) -> np.ndarray:
        return self.forward(waves)

    def infer(self, waves: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw and polynomial-calibrated DNSMOS scores."""
        sbo_raw = self.forward(waves)
        p2 = np.asarray([-0.08397278, -0.13166888, -0.06766283],
                        dtype=np.float32)[None, :]
        p1 = np.asarray([1.22083953, 1.60915514, 1.11546468],
                        dtype=np.float32)[None, :]
        p0 = np.asarray([0.0052439, -0.39604546, 0.04602535],
                        dtype=np.float32)[None, :]
        sbo = p2 * np.square(sbo_raw) + p1 * sbo_raw + p0
        return sbo_raw, sbo

    def script(self, audio: ArrayLike) -> Dict[str, float]:
        """Run the DNSMOS sliding-window clip evaluation on one waveform."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        len_samples = self.input_samples

        while len(audio) < len_samples:
            audio = np.concatenate((audio, audio))

        num_hops = int(np.floor(len(audio) / self.sample_rate)
                       - self.input_length) + 1
        hop_len_samples = self.sample_rate

        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            start = int(idx * hop_len_samples)
            end = int((idx + self.input_length) * hop_len_samples)
            audio_seg = audio[start:end]
            if len(audio_seg) < len_samples:
                continue

            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.forward(
                audio_seg[None])[0].tolist()
            mos_sig, mos_bak, mos_ovr = get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, False)

            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        return {
            "num_hops": num_hops,
            "OVRL_raw": np.mean(predicted_mos_ovr_seg_raw),
            "SIG_raw": np.mean(predicted_mos_sig_seg_raw),
            "BAK_raw": np.mean(predicted_mos_bak_seg_raw),
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
        }

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.config.items())
