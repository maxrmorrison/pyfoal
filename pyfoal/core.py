import contextlib
import functools
import os

import torchaudio
import tqdm

import pyfoal


###############################################################################
# Forced alignment
###############################################################################


def align(
    text,
    audio,
    sample_rate,
    aligner=pyfoal.DEFAULT_ALIGNER,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Phoneme-level forced-alignment

    Arguments
        text : string
            The speech transcript
        audio : np.array(shape=(samples,))
            The speech signal to process
        sample_rate : int
            The audio sampling rate

    Returns
        alignment : Alignment
            The forced alignment
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        return pyfoal.baselines.mfa.align(text, audio, sample_rate)

    # Penn phonetic forced aligner
    if aligner == 'p2fa':
        return pyfoal.baselines.p2fa.align(text, audio, sample_rate)

    # RAD-TTS neural alignment
    if aligner == 'radtts':

        # TODO
        pass

    raise ValueError(f'Aligner {aligner} is not defined')


def from_file(
    text_file,
    audio_file,
    aligner=pyfoal.DEFAULT_ALIGNER,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Phoneme alignment from audio and text files

    Arguments
        text_file : Path
            The corresponding transcript file
        audio_file : Path
            The audio file to process
        aligner : str
            The alignment method to use
        checkpoint : Path
            The checkpoint to use for neural methods
        gpu : int
            The index of the gpu to perform alignment on for neural methods

    Returns
        alignment : Alignment
            The forced alignment
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        return pyfoal.baselines.mfa.from_file(text_file, audio_file)

    # Penn phonetic forced aligner
    if aligner == 'p2fa':
        return pyfoal.baselines.p2fa.from_file(text_file, audio_file)

    # RAD-TTS neural alignment
    if aligner == 'radtts':

        # Load text
        text = pyfoal.load.text(text_file)

        # Load audio
        audio, sample_rate = pyfoal.load.audio(audio_file)

        # Align
        return align(text, audio, sample_rate, aligner, checkpoint, gpu)

    raise ValueError(f'Aligner {aligner} is not defined')


def from_file_to_file(
    text_file,
    audio_file,
    output_file,
    aligner=pyfoal.DEFAULT_ALIGNER,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform phoneme alignment from files and save to disk

    Arguments
        text_file : Path
            The corresponding transcript file
        audio_file : Path
            The audio file to process
        output_file : Path
            The file to save the alignment
        aligner : str
            The alignment method to use
        checkpoint : Path
            The checkpoint to use for neural methods
        gpu : int
            The index of the gpu to perform alignment on for neural methods
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        pyfoal.baselines.mfa.from_file_to_file(
            text_file,
            audio_file,
            output_file)

    # Penn phonetic forced aligner
    elif aligner == 'p2fa':
        pyfoal.baselines.p2fa.from_file_to_file(
            text_file,
            audio_file,
            output_file)

    # RAD-TTS neural alignment
    elif aligner == 'radtts':

        # Align
        alignment = from_file(text_file, audio_file, aligner, checkpoint, gpu)

        # Save
        alignment.save(output_file)

    raise ValueError(f'Aligner {aligner} is not defined')


def from_files_to_files(
    text_files,
    audio_files,
    output_files,
    aligner=pyfoal.DEFAULT_ALIGNER,
    num_workers=None,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform parallel phoneme alignment from many files and save to disk

    Arguments
        text_files : list
            The transcript files
        audio_files : list
            The corresponding speech audio files
        output_files : list
            The files to save the alignments
        aligner : str
            The alignment method to use
        num_workers : int
            Number of CPU cores to utilize. Defaults to all cores.
        checkpoint : Path
            The checkpoint to use for neural methods
        gpu : int
            The index of the gpu to perform alignment on for neural methods
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        pyfoal.baselines.mfa.from_files_to_files(
            text_files,
            audio_files,
            output_files,
            num_workers)

    # Penn phonetic forced aligner
    elif aligner == 'p2fa':
        pyfoal.baselines.p2fa.from_files_to_files(
            text_files,
            audio_files,
            output_files,
            num_workers)

    # RAD-TTS neural alignment
    elif aligner == 'radtts':
        align_fn = functools.partial(
            from_file_to_file,
            checkpoint=checkpoint,
            gpu=gpu)
        for item in zip(text_files, audio_files, output_files):
            align_fn(*item)

    raise ValueError(f'Aligner {aligner} is not defined')


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def chdir(directory):
    """Context manager for changing the current working directory"""
    previous_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous_directory)


def iterator(iterable, message, initial=0, total=None):
    """Create a tqdm iterator"""
    total = len(iterable) if total is None else total
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=total)


def resample(audio, sample_rate, target_rate=pyfoal.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
