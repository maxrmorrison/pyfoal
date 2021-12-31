import contextlib
import functools
import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path

import pypar
import resampy
import soundfile

import pyfoal


###############################################################################
# Constants
###############################################################################


# The aligner to use. One of ['htk', 'mfa'].
ALIGNER = 'mfa'

# The location of the aligner model and phoneme dictionary
ASSETS_DIR = Path(__file__).parent / 'assets' / ALIGNER

# The default audio sampling rate of HTK
HTK_SAMPLE_RATE = 11025


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
    if ALIGNER == 'htk':
        duration = len(audio) / sample_rate

        # Maybe resample
        if sample_rate != HTK_SAMPLE_RATE:
            resampy.resample(audio, sample_rate, HTK_SAMPLE_RATE)

        # Cache aligner
        if not hasattr(align, 'aligner'):
            align.aligner = pyfoal.p2fa.Aligner()

        # Perform forced alignment
        return align.aligner(text, audio, duration)
    elif ALIGNER == 'mfa':

        # Write to temporary storage
        with tempfile.TemporaryDirectory() as directory:
            with chdir(directory):
                soundfile.write('audio.wav', audio, sample_rate)
                with open('text.txt') as file:
                    file.write(text)

                # Align
                from_files_to_files(
                    [Path('text.txt')],
                    [Path('audio.wav')],
                    [Path('alignment.TextGrid')])

                # Load
                return pypar.Alignment('alignment.TextGrid')
    else:
        raise ValueError(f'Aligner {ALIGNER} is not defined')


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
    if ALIGNER == 'htk':

        # Launch multiprocessed HTK alignment
        with mp.get_context('spawn').Pool(num_workers) as pool:
            align_fn = functools.partial(from_file_to_file)
            pool.starmap(align_fn, zip(text_files, audio_files, output_files))

    elif ALIGNER == 'mfa':

        import montreal_forced_aligner as mfa

        # Download english dictionary and acoustic model
        mfa.command_line.model.download_model('dictionary', 'english')
        mfa.command_line.model.download_model('acoustic', 'english')

        with tempfile.TemporaryDirectory() as directory:

            # Copy files to temporary directory, preserving speaker
            for audio_file, text_file in zip(audio_files, text_files):
                shutil.copyfile(
                    audio_file,
                    directory / audio_file.parent.name / audio_file.name)
                shutil.copyfile(
                    text_file,
                    directory / text_file.parent.name / text_file.name)

            # Align
            aligner = mfa.alignment.PretrainedAligner(
                corpus_directory=directory,
                dictionary_path='english',
                acoustic_model_path='english',
                num_jobs=num_workers)
            aligner.align()

            # Copy alignments to destination
            for audio_file, output_file in zip(audio_files, output_files):
                textgrid_file = (
                    directory /
                    audio_file.parent.name /
                    f'{audio_file.stem}.TextGrid')
                shutil.copyfile(textgrid_file, output_file)
    else:
        raise ValueError(f'Aligner {ALIGNER} is not defined')


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def chdir(directory):
    """Context manager for changing the current working directory"""
    curr_dir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curr_dir)
