import functools

import pypar

import pyfoal


###############################################################################
# RAD-TTS neural aligner
###############################################################################


def align(text, audio, sample_rate):
    """Align text and audio using a RAD-TTS neural alignment model"""
    # TODO
    pass


def from_file(text_file, audio_file):
    """Align text and audio on disk using RAD-TTS"""
    # Load text
    text = pyfoal.load.text(text_file)

    # Load audio
    audio, sample_rate = pyfoal.load.audio(audio_file)

    # Align
    return align(text, audio, sample_rate)


def from_file_to_file(text_file, audio_file, output_file):
    """Align text and audio on disk using RAD-TTS and save"""
    from_file(text_file, audio_file).save(output_file)


def from_files_to_files(
    text_files,
    audio_files,
    output_files,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Align text and audio on disk using RAD-TTS and save"""
    align_fn = functools.partial(
        from_file_to_file,
        checkpoint=checkpoint,
        gpu=gpu)
    for item in zip(text_files, audio_files, output_files):
        align_fn(*item)
