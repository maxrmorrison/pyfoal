import string

import g2p_en
import torch

import pyfoal


###############################################################################
# Carnegie Mellon (CMU) G2P
###############################################################################


def from_text(text):
    """Convert text to cmu"""
    # Remove newlines, tabs, and extra whitespace
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')

    # Convert numbers to text
    text = g2p_en.expand.normalize_numbers(text)

    # Remove punctuation
    punctuation = [s for s in string.punctuation + '”“—' if s != '-']
    text = text.translate(str.maketrans('-', ' ', ''.join(punctuation)))

    # Grapheme-to-phoneme conversion
    phonemes = g2p_en.G2p()(text)

    # Convert to indices
    indices = pyfoal.convert.phonemes_to_indices(phonemes)

    return torch.tensor(indices, dtype=torch.long)
