import warnings

import librosa
import numpy as np
import torch

import pyfoal


###############################################################################
# Constants
###############################################################################


# Minimum decibel level
MIN_DB = -100.

# Reference decibel level
REF_DB = 20.


###############################################################################
# A-weighted loudness
###############################################################################


def from_audio(audio):
    """Retrieve the per-frame loudness"""
    # Save device
    device = audio.device

    # Pad
    padding = (pyfoal.WINDOW_SIZE - pyfoal.HOPSIZE) // 2
    audio = torch.nn.functional.pad(audio, (padding, padding))

    # Convert to numpy
    audio = audio.detach().cpu().numpy().squeeze(0)

    # Cache weights
    if not hasattr(from_audio, 'weights'):
        from_audio.weights = perceptual_weights()

    # Take stft
    stft = librosa.stft(
        audio,
        n_fft=pyfoal.WINDOW_SIZE,
        hop_length=pyfoal.HOPSIZE,
        win_length=pyfoal.WINDOW_SIZE,
        center=False)

    # Compute magnitude on db scale
    db = librosa.amplitude_to_db(np.abs(stft))

    # Apply A-weighting
    weighted = db + from_audio.weights

    # Threshold
    weighted[weighted < MIN_DB] = MIN_DB

    # Average over weighted frequencies
    return torch.from_numpy(weighted.mean(axis=0)).float().to(device)[None]


def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(
        sr=pyfoal.SAMPLE_RATE,
        n_fft=pyfoal.WINDOW_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - REF_DB
    