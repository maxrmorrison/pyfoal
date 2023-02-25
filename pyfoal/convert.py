import warnings

import numpy as np
import pypar

import pyfoal


###############################################################################
# Index conversions
###############################################################################


def alignment_to_indices(
    alignment,
    hopsize,
    return_word_breaks=False,
    times=None):
    """Convert alignment to framewise phoneme indices"""
    # Get phonemes at regular time intervals
    if times is None:
        times = np.arange(0, alignment.duration() - 1e-10, hopsize)
    phonemes = [alignment.phoneme_at_time(time) for time in times]

    # Convert to integers
    indices = [phoneme_to_index(p) for p in phonemes]

    # Maybe return word break locations to recover alignment
    if return_word_breaks:
        word_breaks = [(str(word), word.start()) for word in alignment]
        return indices, word_breaks

    return indices


def index_to_phoneme(index):
    """Convert integer index representing a phoneme to a string"""
    # Cache map
    if not hasattr(index_to_phoneme, 'map'):
        index_to_phoneme.map = {
            i: phoneme for i, phoneme in enumerate(pyfoal.load.phonemes())}

    # Convert
    return index_to_phoneme.map[index]


def indices_to_phonemes(indices):
    """Convert index sequence to phoneme sequence"""
    return [index_to_phoneme(index) for index in indices.tolist()]


def indices_to_alignment(indices, hopsize, word_breaks=None):
    """Convert framewise phoneme indices to a phoneme alignment"""
    # If no word breaks are given, populate an empty word
    if word_breaks is None:
        word_breaks = [('', 0.)]

    # Determine word split indices
    word_break_indices = [
        (word, int(time / hopsize)) for word, time in word_breaks]

    # Iterate over frames
    j = 0
    start = 0.
    previous = -1
    words, phonemes = [], []
    for i, index in enumerate(indices):

        if previous != -1:

            # Maybe start a new word
            if j < len(word_breaks) - 1 and i == word_break_indices[j + 1][1]:
                end = i * hopsize
                phonemes.append(
                    pypar.Phoneme(index_to_phoneme(previous), start, end))
                words.append(pypar.Word(word_break_indices[j][0], phonemes))
                start = end
                j += 1

            # Maybe start a new phoneme
            elif index != previous:
                end = i * hopsize
                phonemes.append(
                    pypar.Phoneme(index_to_phoneme(previous), start, end))
                start = end

        previous = index

    # Write last word
    end = len(indices) / hopsize
    phonemes.append(pypar.Phoneme(index_to_phoneme(previous), start, end))
    words.append(pypar.Word(word_break_indices[-1][0], phonemes))

    return pypar.Alignment(words)


def phoneme_to_index(phoneme):
    """Convert phoneme as a string to an integer index"""
    # Cache map
    if not hasattr(phoneme_to_index, 'map'):
        phoneme_to_index.map = {
            phoneme: i for i, phoneme in enumerate(pyfoal.load.phonemes())}

    # Convert
    return phoneme_to_index.map[str(phoneme)]


def phonemes_to_indices(phonemes):
    """Convert phoneme sequence to index sequence"""
    return [phoneme_to_index(phoneme) for phoneme in phonemes]


###############################################################################
# Time conversions
###############################################################################


def seconds_to_frames(seconds):
    """Convert seconds to frames"""
    return int(seconds * pyfoal.SAMPLE_RATE / pyfoal.HOPSIZE)


def frames_to_samples(frames):
    """Convert number of frames to samples"""
    return frames * pyfoal.HOPSIZE


def frames_to_seconds(frames):
    """Convert number of frames to seconds"""
    return frames * samples_to_seconds(pyfoal.HOPSIZE)


def samples_to_seconds(samples, sample_rate=pyfoal.SAMPLE_RATE):
    """Convert time in samples to seconds"""
    return samples / sample_rate


def samples_to_frames(samples):
    """Convert time in samples to frames"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return samples // pyfoal.HOPSIZE
    