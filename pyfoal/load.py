import csv

import pyfoal


###############################################################################
# Loading
###############################################################################


def phonemes():
    """Load list of phonemes"""
    # Cache phonemes
    if not hasattr(phonemes, pyfoal.ALIGNER):
        with open(pyfoal.ASSETS_DIR / pyfoal.ALIGNER / 'phonemes.csv') as file:
            reader = csv.reader(file)

            # Skip header
            next(reader)

            # Update cache
            setattr(phonemes, pyfoal.ALIGNER, [row[0] for row in reader])

    # Get phonemes
    return getattr(phonemes, pyfoal.ALIGNER)


def voicing():
    """Load a map that indicates whether each phoneme is voiced"""
    # Cache voicing
    if not hasattr(voicing, 'voicing'):
        with open(pyfoal.ASSETS_DIR / pyfoal.ALIGNER / 'phonemes.csv') as file:
            reader = csv.reader(file)

            # Skip header
            next(reader)

            # Get voicing
            voicing.voicing = {row[0]: row[1] for row in reader}
    return voicing.voicing


def vowels():
    """Load a map that indicates whether each phoneme is a vowel"""
    # Cache vowels
    if not hasattr(vowels, 'vowels'):
        with open(pyfoal.ASSETS_DIR / pyfoal.ALIGNER / 'phonemes.csv') as file:
            reader = csv.reader(file)

            # Skip header
            next(reader)

            # Get vowels
            vowels.vowels = {row[0]: row[2] for row in reader}
    return vowels.vowels
