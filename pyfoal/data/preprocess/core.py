import numpy as np
import scipy
import torch

import pyfoal


###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess a dataset
    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        directory = pyfoal.CACHE_DIR / dataset

        # Get text files
        text_files = list(directory.rglob('*.txt'))

        # Get output phoneme files
        phoneme_files = [file.with_suffix('.pt') for file in text_files]

        # Grapheme-to-phoneme
        pyfoal.g2p.from_files_to_files(text_files, phoneme_files)


def prior(phoneme_length, frame_length):
    """Beta-binomial attention prior"""
    indices = np.arange(0, phoneme_length)
    priors = []
    for i in range(1, frame_length + 1):
        a = pyfoal.ATTENTION_PRIOR_SCALE_FACTOR * i
        b = pyfoal.ATTENTION_PRIOR_SCALE_FACTOR * (frame_length - i + 1)
        prior = scipy.stats.betabinom(phoneme_length - 1, a, b).pmf(indices)
        priors.append(prior)
    return torch.tensor(np.array(priors))
