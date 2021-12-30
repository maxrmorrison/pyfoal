import copy

import pyfoal


###############################################################################
# Phoneme alignment interpolation
###############################################################################


def phonemes(alignment, ratio):
    """Interpolate the alignment by uniformly interpolating all phonemes"""
    alignment = copy.deepcopy(alignment)
    durations = [(1. / ratio) * p.duration() for p in alignment.phonemes()]
    alignment.update(durations=durations)
    return alignment


def voiced(alignment, ratio):
    """Interpolate the alignment by interpolating only voiced regions"""
    alignment = copy.deepcopy(alignment)

    # Determine which phonemes to stretch
    phonemes = alignment.phonemes()
    voicing = [is_voiced(str(phoneme)) for phoneme in phonemes]

    # Compute the ratio of the stretch applied to voiced regions
    duration = alignment.duration()
    target_duration = ratio * duration
    duration_voiced = sum([
        p.duration() for p, voiced in zip(phonemes, voicing) if voiced])
    duration_unvoiced = duration - duration_voiced
    voiced_ratio = (target_duration - duration_unvoiced) / duration_voiced

    # Compute new durations
    durations = []
    for phoneme, voiced in zip(phonemes, voicing):
        duration = phoneme.duration()
        durations.append(voiced_ratio * duration if voiced else duration)

    # Update the alignment
    alignment.update(durations=durations)

    return alignment


def vowels(alignment, ratio):
    """Interpolate the alignment by interpolating only vowels"""
    alignment = copy.deepcopy(alignment)

    # Determine which phonemes to stretch
    phonemes = alignment.phonemes()
    vowels = [is_vowel(str(phoneme)) for phoneme in phonemes]

    # Compute the ratio of the stretch applied to voiced regions
    duration = alignment.duration()
    target_duration = ratio * duration
    duration_vowel = sum([
        p.duration() for p, vowel in zip(phonemes, vowels) if vowel])
    duration_consonant = duration - duration_vowel
    vowel_ratio = (target_duration - duration_consonant) / duration_vowel

    # Compute new durations
    durations = []
    for phoneme, vowel in zip(phonemes, vowels):
        duration = phoneme.duration()
        durations.append(vowel_ratio * duration if vowel else duration)

    # Update the alignment
    alignment.update(durations=durations)

    return alignment


###############################################################################
# Utilities
###############################################################################


def is_voiced(phoneme):
    """Returns True iff the phoneme is voiced"""
    return bool(pyfoal.load.voicing()[phoneme])


def is_vowel(phoneme):
    """Returns True iff the phoneme is a vowel"""
    return bool(pyfoal.load.vowels()[phoneme])
