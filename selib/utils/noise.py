import math
import random
from typing import List, Literal

import numpy as np


NOISE_TYPE = Literal[
    "white",
    "pink",
    "brown",
    "blue",
    "violet",
    "bandpass",
    "hum",
    "impulse",
]
COLORED_NOISE = Literal["pink", "brown", "blue", "violet"]


class NoiseGenerator:
    """Generate artificial mono noise waveforms as NumPy arrays.

    Generated arrays have shape [n_samples] and dtype float32 by default.
    They are RMS-normalized unless rms=None is passed.
    """

    default_noise_types = (
        "white",
        "pink",
        "brown",
        "blue",
        "violet",
        "bandpass",
        "hum",
        "impulse",
    )

    def __init__(self,
                 sample_rate: int = 16000,
                 noise_types: List[NOISE_TYPE] = None,
                 rms: float = 1.0,
                 seed: int = None,
                 dtype=np.float32):
        self.sample_rate = sample_rate
        self.noise_types = tuple(noise_types or self.default_noise_types)
        self.rms = rms
        self.dtype = dtype

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def sample(self,
               n_samples: int,
               noise_type: NOISE_TYPE = None,
               rms: float = None) -> np.ndarray:
        noise_type = noise_type or self.rng.choice(self.noise_types)
        noise_type = noise_type.lower()

        if noise_type == "white":
            noise = self._white(n_samples)
        elif noise_type in ("pink", "brown", "blue", "violet"):
            noise = self._colored(n_samples, noise_type)
        elif noise_type == "bandpass":
            noise = self._bandpass(n_samples)
        elif noise_type == "hum":
            noise = self._hum(n_samples)
        elif noise_type == "impulse":
            noise = self._impulse(n_samples)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        noise = self._normalize(noise, self.rms if rms is None else rms)
        return noise.astype(self.dtype, copy=False)

    def sample_batch(self,
                     batch_size: int,
                     n_samples: int,
                     noise_type: NOISE_TYPE = None,
                     rms: float = None) -> np.ndarray:
        noises = [
            self.sample(n_samples, noise_type=noise_type, rms=rms)
            for _ in range(batch_size)
        ]
        return np.stack(noises, axis=0)

    def __call__(self,
                 n_samples: int,
                 noise_type: NOISE_TYPE = None,
                 rms: float = None) -> np.ndarray:
        return self.sample(n_samples, noise_type=noise_type, rms=rms)

    def _white(self, n_samples: int) -> np.ndarray:
        return self.np_rng.standard_normal(n_samples)

    def _colored(self,
                 n_samples: int,
                 color: COLORED_NOISE) -> np.ndarray:
        white = self._white(n_samples)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1 / self.sample_rate)

        scale = np.ones_like(freqs)
        scale[0] = 0.0
        nonzero = freqs > 0

        if color == "pink":
            scale[nonzero] = freqs[nonzero] ** -0.5
        elif color == "brown":
            scale[nonzero] = freqs[nonzero] ** -1.0
        elif color == "blue":
            scale[nonzero] = freqs[nonzero] ** 0.5
        elif color == "violet":
            scale[nonzero] = freqs[nonzero]

        return np.fft.irfft(spec * scale, n=n_samples)

    def _bandpass(self, n_samples: int) -> np.ndarray:
        white = self._white(n_samples)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1 / self.sample_rate)

        low = self.rng.uniform(80, min(3500, self.sample_rate / 4))
        high = self.rng.uniform(low * 1.5, min(7600, self.sample_rate / 2))
        mask = (freqs >= low) & (freqs <= high)
        return np.fft.irfft(spec * mask, n=n_samples)

    def _hum(self, n_samples: int) -> np.ndarray:
        t = np.arange(n_samples) / self.sample_rate
        base_freq = self.rng.choice((50.0, 60.0))
        base_freq *= self.rng.uniform(0.98, 1.02)
        n_harmonics = self.rng.randint(2, 6)

        noise = np.zeros(n_samples)
        for harmonic in range(1, n_harmonics + 1):
            freq = base_freq * harmonic
            if freq >= self.sample_rate / 2:
                break
            phase = self.rng.uniform(0, 2 * math.pi)
            amp = 1.0 / harmonic
            noise = noise + amp * np.sin(2 * math.pi * freq * t + phase)

        return noise + 0.03 * self._white(n_samples)

    def _impulse(self, n_samples: int) -> np.ndarray:
        noise = 0.02 * self._white(n_samples)
        density = self.rng.uniform(0.0005, 0.005)
        n_impulses = max(1, int(n_samples * density))

        idx = self.np_rng.integers(0, n_samples, size=n_impulses)
        amps = self.np_rng.standard_normal(n_impulses)
        np.add.at(noise, idx, amps)
        return noise

    @staticmethod
    def _normalize(noise: np.ndarray, rms: float) -> np.ndarray:
        if rms is None:
            return noise
        noise_rms = np.sqrt(np.mean(np.square(noise)))
        noise_rms = max(noise_rms, 1e-12)
        return noise * (rms / noise_rms)
