# SeLib

SeLib is a small speech-enhancement library for ONNX models. It provides a
simple `enhance()` function, lightweight model wrappers, and utility functions for audio/noise experiments.

## Install

Install the package directly from GitHub for the latest version.

```bash
pip install git+https://github.com/nipponjo/selib.git
```

or the latest release:

```bash
pip install speech-selib
```

## Basic Use

The shortest path is `selib.enhance(wave, model_id)`. Pass a mono NumPy waveform at the sample rate expected by the model.
```python
import librosa
import selib

from selib.utils import NoiseGenerator, snr_mixer

# Load clean speech at the model sample rate.
wave_clean, sr = librosa.load("./data/clean_freesound_33711.wav", sr=48000)

# Create artificial noise and mix it with the clean speech at 5 dB SNR.
noise_gen = NoiseGenerator(sample_rate=sr)
noise = noise_gen.sample(n_samples=len(wave_clean))
wave_noisy = snr_mixer(wave_clean, noise, snr=5)

# Enhance with a registered model id.
# The ONNX model is downloaded and cached automatically on first use.
wave_enhanced = selib.enhance(wave_noisy, "deepfilternet3")
```

You can also save the enhanced waveform directly. The loaded model is cached by
default, so repeated calls with the same `model_id` do not reload the ONNX
session.

```python
# Save as 24-bit PCM WAV while returning the enhanced NumPy array.
wave_enhanced = selib.enhance(
    wave_noisy,
    "deepfilternet3",    
    save_to="enhanced.wav",
    bits_per_sample=24,
)
```

## Model Objects

For more control, load the model wrapper once and call its
`enhance()` method directly.

```python
from selib import load_model
from selib.models import DeepFilterNetOnnx

# load_model() chooses the correct wrapper from the registry metadata.
model: DeepFilterNetOnnx = load_model("deepfilternet3")

# DeepFilterNetOnnx supports extra options such as attenuation limiting.
wave_enhanced = model.enhance(wave_noisy, atten_lim_db=12)
```

Magnitude-mask models can also be used directly. They predict a mask, multiply
it with the noisy magnitude spectrogram, and reuse the noisy phase.

```python
from selib.models import MagnitudeMaskModel

# ul_unas_16k is a 16 kHz magnitude-mask model.
mask_model = MagnitudeMaskModel("ul_unas_16k")
wave_enhanced = mask_model.enhance(wave_noisy_16k)
```

## Metrics

SeLib includes a few quick NumPy-only metrics that do not require PESQ/STOI or
other external scoring libraries.

```python
from selib.metrics import snr, segsnr, si_sdr

# Compare clean reference speech against an enhanced waveform.
print("SNR:", snr(wave_clean, wave_enhanced))
print("segSNR:", segsnr(wave_clean, wave_enhanced, sample_rate=sr))
print("SI-SDR:", si_sdr(wave_clean, wave_enhanced))
```

## Available Models

| Model ID | Type | Sample rate | #params | Paper | Repository |
| --- | --- | --- | --- | --- | --- |
| `deepfilternet3` | DeepFilterNet | 48 kHz | 2.13M | [arXiv:2305.08227](https://arxiv.org/abs/2305.08227) | [GitHub](https://github.com/rikorose/deepfilternet) |
| `deepfilternet2` | DeepFilterNet | 48 kHz | 2.31M | [arXiv:2205.05474](https://arxiv.org/abs/2205.05474) | [GitHub](https://github.com/rikorose/deepfilternet) |
| `deepfilternet1` | DeepFilterNet | 48 kHz | 1.78M | [arXiv:2110.05588](https://arxiv.org/abs/2110.05588) | [GitHub](https://github.com/rikorose/deepfilternet) |
| `ul_unas_16k` | Magnitude mask | 16 kHz | 0.171M | [arXiv:2503.00340](https://arxiv.org/abs/2503.00340) | [GitHub](https://github.com/Xiaobin-Rong/ul-unas) |

DeepFilterNet models use ERB masks plus deep-filter coefficients. Magnitude-mask
models predict a spectrogram mask and reconstruct audio with the noisy phase.
