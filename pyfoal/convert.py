import itertools

import pypar

import pyfoal


###############################################################################
# Conversion functions
###############################################################################


def phoneme_to_index(phoneme):
    """Convert phoneme as a string to an integer index"""
    # Cache map
    if not hasattr(phoneme_to_index, 'map'):
        phoneme_to_index.map = {
            phoneme: i for i, phoneme in enumerate(pyfoal.load.phonemes())}

    # Convert
    return phoneme_to_index.map[str(phoneme)]


def index_to_phoneme(index):
    """Convert integer index representing a phoneme to a string"""
    # Cache map
    if not hasattr(index_to_phoneme, 'map'):
        index_to_phoneme.map = {
            i: phoneme for i, phoneme in enumerate(pyfoal.load.phonemes())}

    # Convert
    return index_to_phoneme.map[index]


def indices_to_alignment(indices, hopsize):
    """Convert framewise phoneme indices to a phoneme alignment"""
    # Get consecutive index identities and times in seconds
    groups = [
        (index, hopsize * sum(1 for _ in group))
        for index, group in itertools.groupby(indices)]

    # Create phonemes
    start = 0.
    phonemes = []
    for index, duration in groups:
        phonemes.append(
            pypar.Phoneme(index_to_phoneme(index)),
            start,
            start + duration)
        start += duration

    # We don't have word break information, so we assume one word
    return pypar.Alignment([pypar.Word(phonemes)])

