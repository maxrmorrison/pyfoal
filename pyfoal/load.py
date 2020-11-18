import torchaudio

import pyfoal


###############################################################################
# Loading
###############################################################################


def audio(file):
    """Load audio file from disk
    Arguments
        file : string
            The audio file to load
    Returns
        audio : torch.tensor(shape=(1, time))
            The audio sampled at 16 kHz
    """
    audio, orig_sample_rate = torchaudio.load(file)

    # Resample
    return torchaudio.transforms.Resample(
        orig_sample_rate, pyfoal.SAMPLE_RATE)(audio)


def text(filename):
    """Load text file from disk
    Arguments
        filename : string
            The text file to load
    Returns
        text : string
            The text
    """
    with open(filename) as file:
        return file.read()
