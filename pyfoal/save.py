import torchaudio

import pyfoal


###############################################################################
# Saving
###############################################################################


def audio(file, audio):
    """Load audio file from disk
    Arguments
        file : string
            The audio file to load
        audio : torch.tensor(shape=(1, samples))
            The audio to save
    """
    torchaudio.save(audiofile, audio, pyfoal.SAMPLE_RATE)
