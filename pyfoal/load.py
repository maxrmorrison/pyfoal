import soundfile

import pyfoal


###############################################################################
# Loading
###############################################################################


def audio(file):
    """Load audio from file"""
    audio, sample_rate = soundfile.read(file)

    # Resample
    return pyfoal.resample(audio, sample_rate)


def phonemes():
    """Load list of phonemes"""
    # Cache phonemes
    if not hasattr(phonemes, 'phonemes'):
        with open(pyfoal.ASSETS_DIR / 'monophones') as file:
            phonemes.phonemes = file.readlines()

    return phonemes.phonemes
