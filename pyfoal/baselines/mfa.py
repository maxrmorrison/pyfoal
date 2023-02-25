import contextlib
import logging
import multiprocessing as mp
import shutil
import tempfile
import warnings
from pathlib import Path

import pypar
import soundfile

import pyfoal


###############################################################################
# Montreal forced alignment
###############################################################################


def from_text_and_audio(text, audio, sample_rate=pyfoal.SAMPLE_RATE):
    """Align text and audio using MFA"""
    # Write to temporary storage
    with tempfile.TemporaryDirectory() as directory:
        with pyfoal.chdir(directory):
            soundfile.write('item.wav', audio.squeeze().numpy(), sample_rate)
            with open('item.txt', 'w') as file:
                file.write(text)

            # Align
            from_files_to_files(
                [Path('item.txt').resolve()],
                [Path('item.wav').resolve()],
                [Path('item.TextGrid').resolve()])

            # Load
            return pypar.Alignment('item.TextGrid')


def from_file(text_file, audio_file):
    """Align text and audio on disk using MFA"""
    # Load text
    text = pyfoal.load.text(text_file)

    # Load audio
    audio, sample_rate = pyfoal.load.audio(audio_file)

    # Align
    return from_text_and_audio(text, audio, sample_rate)


def from_file_to_file(text_file, audio_file, output_file):
    """Align text and audio on disk using MFA and save"""
    from_file(text_file, audio_file).save(output_file)


def from_files_to_files(
    text_files,
    audio_files,
    output_files,
    num_workers=None):
    """Align text and audio on disk using MFA and save"""
    import montreal_forced_aligner as mfa

    # Download english dictionary and acoustic model
    manager = mfa.models.ModelManager()
    manager.download_model('dictionary', 'english_mfa')
    manager.download_model('acoustic', 'english_mfa')

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Copy files to temporary directory, preserving speaker
        with mp.Pool(num_workers) as pool:
            iterator = zip(
                [directory] * len(text_files),
                text_files,
                audio_files)
            pool.starmap(copy_and_convert, iterator)

        # MFA generates a lot of log information we don't need
        with disable_logging(logging.CRITICAL):

            # Setup aligner
            aligner = mfa.alignment.PretrainedAligner(
                corpus_directory=str(directory),
                dictionary_path='english_mfa',
                acoustic_model_path='english_mfa',
                num_jobs=num_workers,
                debug=False,
                verbose=False)

            # Align
            aligner.align()

            # Export alignments
            aligner.export_files(str(directory))

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
                warnings.warn('MFA failed to align at least one file')


###############################################################################
# Utilities
###############################################################################


def copy_and_convert(directory, text_file, audio_file):
    """Prepare text and audio files for MFA alignment"""
    speaker_directory = directory / audio_file.parent.name
    speaker_directory.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(text_file, speaker_directory / text_file.name)

    # Aligning fails if the audio is not a 16-bit mono wav file, so
    # we convert instead of copy
    audio, sample_rate = soundfile.read(str(audio_file))
    soundfile.write(
        str(speaker_directory / f'{audio_file.stem}.wav'),
        audio.squeeze(),
        sample_rate)


@contextlib.contextmanager
def disable_logging(level):
    """Context manager for changing the current log level of all loggers"""
    try:
        logging.disable(level)
        yield
    finally:
        logging.disable(logging.NOTSET)
