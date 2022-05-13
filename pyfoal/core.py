import contextlib
import functools
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import pypar
import resampy
import soundfile

import pyfoal


###############################################################################
# Forced alignment
###############################################################################


def align(text, audio, sample_rate):
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
    if pyfoal.ALIGNER == 'p2fa':
        duration = len(audio) / sample_rate

        # Maybe resample
        if sample_rate != pyfoal.p2fa.SAMPLE_RATE:
            resampy.resample(audio, sample_rate, pyfoal.p2fa.SAMPLE_RATE)

        # Cache aligner
        if not hasattr(align, 'aligner'):
            align.aligner = pyfoal.p2fa.Aligner()

        # Perform forced alignment
        return align.aligner(text, audio, duration)
    elif pyfoal.ALIGNER == 'mfa':

        # Write to temporary storage
        with tempfile.TemporaryDirectory() as directory:
            with chdir(directory):
                soundfile.write('item.wav', audio, sample_rate)
                with open('item.txt', 'w') as file:
                    file.write(text)

                # Align
                from_files_to_files(
                    [Path('item.txt').resolve()],
                    [Path('item.wav').resolve()],
                    [Path('item.TextGrid').resolve()])

                # Load
                return pypar.Alignment('item.TextGrid')
    else:
        raise ValueError(f'Aligner {pyfoal.ALIGNER} is not defined')


def from_file(text_file, audio_file):
    """Phoneme alignment from audio and text files

    Arguments
        text_file : Path
            The corresponding transcript file
        audio_file : Path
            The audio file to process

    Returns
        alignment : Alignment
            The forced alignment
    """
    # Load text
    with open(text_file) as file:
        text = file.read()

    # Load audio
    audio, sample_rate = soundfile.read(audio_file)

    # Align
    return align(text, audio, sample_rate)


def from_file_to_file(text_file, audio_file, output_file):
    """Perform phoneme alignment from files and save to disk

    Arguments
        text_file : Path
            The corresponding transcript file
        audio_file : Path
            The audio file to process
        output_file : Path
            The file to save the alignment
    """
    # Align and save
    from_file(text_file, audio_file).save(output_file)


def from_files_to_files(
    text_files,
    audio_files,
    output_files,
    num_workers=None):
    """Perform parallel phoneme alignment from many files and save to disk

    Arguments
        text_files : list
            The transcript files
        audio_files : list
            The corresponding speech audio files
        output_files : list
            The files to save the alignments
        num_workers : int
            Number of CPU cores to utilize. Defaults to all cores.
    """
    # Default to using all cpus
    num_workers = num_workers if num_workers else os.cpu_count()

    if pyfoal.ALIGNER == 'p2fa':

        # Launch multiprocessed P2FA alignment
        with mp.Pool(num_workers) as pool:
            align_fn = functools.partial(from_file_to_file)
            pool.starmap(align_fn, zip(text_files, audio_files, output_files))

    elif pyfoal.ALIGNER == 'mfa':

        try:
            import montreal_forced_aligner as mfa

            # Download english dictionary and acoustic model
            mfa.command_line.model.download_model('dictionary', 'english')
            mfa.command_line.model.download_model('acoustic', 'english')

            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)

                # Copy files to temporary directory, preserving speaker
                with mp.Pool(num_workers) as pool:
                    iterator = zip(
                        [directory] * len(text_files),
                        text_files,
                        audio_files)
                    pool.starmap(mfa_copy_and_convert, iterator)

                # MFA generates a lot of log information we don't need
                with disable_logging(logging.CRITICAL):

                    # Setup aligner
                    aligner = mfa.alignment.PretrainedAligner(
                        corpus_directory=str(directory),
                        dictionary_path='english',
                        acoustic_model_path='english',
                        num_jobs=num_workers,
                        debug=False,
                        verbose=False)

                    # Align
                    aligner.align()

                    # Export alignments
                    aligner.export_files(directory)

                # Copy alignments to destination
                for audio_file, output_file in zip(audio_files, output_files):
                    textgrid_file = (
                        directory /
                        audio_file.parent.name /
                        f'{audio_file.stem}.TextGrid')

                    # The alignment can fail. This typically indicates that the
                    # transcript and audio do not match. We skip these files.
                    try:
                        pypar.Alignment(textgrid_file).save(output_file)
                    except FileNotFoundError:
                        warnings.warn(
                            'MFA failed to align. Maybe retry with P2FA.')

        except Exception as error:
            warnings.warn(f'pyfoal - MFA failed with {error}')
    else:
        raise ValueError(f'Aligner {pyfoal.ALIGNER} is not defined')


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def backend(aligner):
    """Change which forced aligner is used"""
    previous_aligner = getattr(pyfoal, 'ALIGNER')
    try:
        setattr(pyfoal, 'ALIGNER', aligner)
        yield
    finally:
        setattr(pyfoal, 'ALIGNER', previous_aligner)


@contextlib.contextmanager
def chdir(directory):
    """Context manager for changing the current working directory"""
    previous_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous_directory)


def mfa_copy_and_convert(directory, text_file, audio_file):
    """Prepare text and audio files for MFA alignment"""
    speaker_directory = directory / audio_file.parent.name
    speaker_directory.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(text_file, speaker_directory / text_file.name)

    # Aligning fails if the audio is not a 16-bit mono wav file, so
    # we convert instead of copy
    audio, sample_rate = soundfile.read(audio_file)
    soundfile.write(
        speaker_directory / f'{audio_file.stem}.wav',
        audio,
        sample_rate)


@contextlib.contextmanager
def disable_logging(level):
    """Context manager for changing the current log level of all loggers"""
    try:
        logging.disable(level)
        yield
    finally:
        logging.disable(logging.NOTSET)
