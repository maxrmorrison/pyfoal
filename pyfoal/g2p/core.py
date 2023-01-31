import multiprocessing

import torch

import pyfoal


###############################################################################
# Grapheme-to-phoneme (G2P)
###############################################################################


def from_text(text):
    """Convert text to phonemes"""
    if pyfoal.G2P == 'cmu':
        phonemes = pyfoal.g2p.cmu.from_text(text)
    elif pyfoal.G2P == 'ipa':
        phonemes = pyfoal.g2p.ipa.from_text(text)
    else:
        raise ValueError(
            f'Grapheme-to-phoneme method {pyfoal.G2P} is not defined')

    # Convert to torch
    return torch.tensor(
        [pyfoal.convert.phoneme_to_index(phoneme) for phoneme in phonemes],
        dtype=torch.long)


def from_file(text_file):
    """Convert text on disk to phonemes"""
    return from_text(pyfoal.load.text(text_file))


def from_file_to_file(text_file, output_file):
    """Convert text on disk to phonemes and save"""
    torch.save(from_file(text_file), output_file)


def from_files_to_files(text_files, output_files):
    """Convert text on disk to phonemes and save"""
    with multiprocessing.Pool(pyfoal.NUM_WORKERS) as pool:
        pool.starmap(from_file_to_file, zip(text_files, output_files))
