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

        # Get audio files
        audio_files = [file.with_suffix('.wav') for file in text_files]

        # Get output phoneme files
        phoneme_files = [
            file.parent / f'{file.stem}-phonemes.pt' for file in text_files]

        # Grapheme-to-phoneme
        pyfoal.g2p.from_files_to_files(text_files, phoneme_files)

        # Get output prior files
        prior_files = [
            file.parent / f'{file.stem}-prior.pt' for file in text_files]

        # Attention prior
        pyfoal.data.preprocess.prior.from_files_to_files(
            text_files,
            audio_files,
            prior_files)
