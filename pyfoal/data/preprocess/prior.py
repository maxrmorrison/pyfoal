import multiprocessing
import os

import numpy as np
import scipy
import torch
import torchaudio

import pyfoal


###############################################################################
# Attention prior
###############################################################################


def from_lengths(phonemes, frames):
    """Beta-binomial attention prior"""
    indices = np.arange(0, phonemes)
    priors = []
    for i in range(1, frames + 1):
        a = pyfoal.ATTENTION_PRIOR_SCALE_FACTOR * i
        b = pyfoal.ATTENTION_PRIOR_SCALE_FACTOR * (frames - i + 1)
        prior = scipy.stats.betabinom(phonemes - 1, a, b).pmf(indices)
        priors.append(prior)
    return torch.tensor(np.array(priors))


def from_file(text_file, audio_file):
    """Compute prior from files on disk"""
    # Load text
    text = pyfoal.load.text(text_file)

    # Get number of frames without loading audio
    frames = pyfoal.convert.samples_to_frames(
        torchaudio.info(audio_file).num_frames)

    # Compute prior
    return from_lengths(len(pyfoal.g2p.from_text(text)[1]), frames)


def from_file_to_file(text_file, audio_file, output_file):
    """Compute attention prior from files and save"""
    torch.save(from_file(text_file, audio_file), output_file)


def from_files_to_files(text_files, audio_files, output_files):
    """Compute attention priors from files and save"""
    with multiprocessing.Pool(os.cpu_count() // 2) as pool:
        pool.starmap(
            from_file_to_file,
            zip(text_files, audio_files, output_files))