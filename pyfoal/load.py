import csv
import json

import torchaudio

import pyfoal


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return pyfoal.resample(audio, sample_rate)


def partition(dataset):
    """Load partitions for dataset"""
    with open(pyfoal.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)


def phonemes():
    """Load list of phonemes"""
    # Cache phonemes
    if not hasattr(phonemes, 'phonemes'):

        # Load file
        with open(pyfoal.ASSETS_DIR / 'g2p' / 'phonemes.csv') as file:
            reader = csv.reader(file)

            # Skip header
            next(reader)

            # Update cache
            phonemes.phonemes = [row[0] for row in reader]
    return phonemes.phonemes


def text(file):
    """Load text from disk"""
    with open(file) as file:
        return file.read()


def voicing():
    """Load a map that indicates whether each phoneme is voiced"""
    # Cache voicing
    if not hasattr(voicing, 'voicing'):
        with open(pyfoal.ASSETS_DIR / 'g2p' / 'phonemes.csv') as file:
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
        with open(pyfoal.ASSETS_DIR / 'g2p' / 'phonemes.csv') as file:
            reader = csv.reader(file)

            # Skip header
            next(reader)

            # Get vowels
            vowels.vowels = {row[0]: row[2] for row in reader}
    return vowels.vowels
