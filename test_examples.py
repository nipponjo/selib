"""Small runnable examples matching the README snippets.

The cells use synthetic audio so they can run without external test files. In
VS Code, Spyder, or Jupyter-compatible editors, each ``# %%`` block can be run
independently after the setup cell.
"""

# %%
# Setup: create short clean/noisy example waveforms at the model sample rates.
from pathlib import Path

import librosa
import numpy as np
import selib

from selib.utils import NoiseGenerator, snr_mixer


def make_tone(sample_rate: int, seconds: float = 0.5) -> np.ndarray:
    """Create a simple speech-like test waveform."""
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    wave = 0.25 * np.sin(2 * np.pi * 220 * t)
    wave += 0.10 * np.sin(2 * np.pi * 440 * t)
    return wave.astype(np.float32)


# DeepFilterNet models expect 48 kHz audio.
sr48 = 48000
# wave_clean = make_tone(sr48)
wave_clean, _ = librosa.load('./data/clean_freesound_33711.wav', sr=48000)

# Create artificial noise and mix it with the clean signal at 5 dB SNR.
noise_gen = NoiseGenerator(sample_rate=sr48)
noise = noise_gen.sample(n_samples=len(wave_clean))
wave_noisy = snr_mixer(wave_clean, noise, snr=5)

# Magnitude-mask example model expects 16 kHz audio.
sr16 = 16000
wave_clean_16k = librosa.resample(wave_clean, orig_sr=sr48, target_sr=sr16)
wave_noisy_16k = librosa.resample(wave_noisy, orig_sr=sr48, target_sr=sr16)


# %%
# Basic use: enhance with a registered model id.
# The ONNX model is downloaded and cached automatically on first use.
wave_enhanced = selib.enhance(wave_noisy, "deepfilternet3")
print("DeepFilterNet3 output:", wave_enhanced.shape)


# %%
# Save use: write the enhanced waveform as 24-bit PCM WAV.
output_dir = Path("tmp")
output_dir.mkdir(exist_ok=True)

wave_enhanced_saved = selib.enhance(
    wave_noisy,
    "deepfilternet3",
    save_to=output_dir / "enhanced_df3.wav",
    bits_per_sample=24,
)
print("Saved output:", wave_enhanced_saved.shape)


# %%
# Model object use: load once, then call enhance repeatedly.
from selib import load_model
from selib.models import DeepFilterNetOnnx


# load_model() chooses the correct wrapper from the registry metadata.
model: DeepFilterNetOnnx = load_model("deepfilternet3")

# DeepFilterNetOnnx supports extra options such as attenuation limiting.
wave_enhanced_limited = model.enhance(wave_noisy, atten_lim_db=12)
print("Attenuation-limited output:", wave_enhanced_limited.shape)


# %%
# Magnitude-mask model use: ul_unas_16k works on 16 kHz audio.
from selib.models import MagnitudeMaskModel


mask_model = MagnitudeMaskModel("ul_unas_16k")
wave_enhanced_16k = mask_model.enhance(wave_noisy_16k)
print("UL-UNAS output:", wave_enhanced_16k.shape)


# %%
# Metrics use: compare clean reference speech against enhanced waveforms.
from selib.metrics import segsnr, si_sdr, snr


print("Noisy SNR:", snr(wave_clean, wave_noisy))
print("DF3 SNR:", snr(wave_clean, wave_enhanced))
print("DF3 segSNR:", segsnr(wave_clean, wave_enhanced, sample_rate=sr48))
print("DF3 SI-SDR:", si_sdr(wave_clean, wave_enhanced))

print("UL-UNAS SNR:", snr(wave_clean_16k, wave_enhanced_16k))
