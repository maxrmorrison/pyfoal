###############################################################################
# Data directory structure
###############################################################################


# data
# ├── cache
# │   ├── <dataset-0>
# |   |   ├── 0000                 (speaker)
# |   |   |   ├── 000000.TextGrid  (alignment - for evaluation datasets only)
# |   |   |   ├── 000000.txt       (text)
# |   |   |   ├── 000000.wav       (audio)
# |   |   |   ├── 000001.TextGrid
# |   |   |   ├── 000001.txt
# |   |   |   ├── 000001.wav
# |   |   |   ├── ...
# |   |   ├── 0001
# |   |   |   ├── ...
# |   |   ├── ...
# │   ├── <dataset-1>
# |   |   ├── ...
# │   ├── ...
# ├── datasets
# |   ├── <dataset-0>
# |   |   ├── {raw contents}
# |   ├── <dataset-1>
# |   |   ├── ...
# |   ├── ...
# └── sources
#     ├── <dataset-0>
#     |   └── {tarballs and zipfiles}
#     ├── <dataset-1>
#     |   └── ...
#     ├── ...


###############################################################################
# Imports
###############################################################################


import requests
import shutil
import tarfile
from pathlib import Path

import pypar
import torchaudio

import pyfoal


###############################################################################
# Constants
###############################################################################


# Abbreviations of speakers in arctic dataset
ARCTIC_SPEAKERS = ['awb', 'bdl', 'clb', 'jmk', 'ksp', 'rms', 'slt']


###############################################################################
# Download datasets
###############################################################################


def datasets(datasets):
    """Download datasets"""
    for dataset in datasets:
        if dataset == 'arctic':
            arctic()
        elif dataset == 'libritts':
            libritts()
        else:
            raise ValueError(
                f'Dataset downloader for {dataset} not implemented')


###############################################################################
# Individual datasets
###############################################################################


def arctic():
    """Download arctic dataset"""
    # Delete data if it already exists
    data_directory = pyfoal.DATA_DIR / 'arctic'
    if data_directory.exists():
        shutil.rmtree(str(data_directory))

    # Create data directory
    data_directory.mkdir(parents=True)

    # URL format string
    url = (
        'http://festvox.org/cmu_arctic/cmu_arctic/' +
        'packed/cmu_us_{}_arctic-0.95-release.tar.bz2')

    # Download audio data
    iterator = pyfoal.iterator(
        ARCTIC_SPEAKERS,
        'Downloading arctic',
        total=len(ARCTIC_SPEAKERS))
    for speaker in iterator:
        download_tar_bz2(url.format(speaker), data_directory)

    # Setup data directory
    cache_directory = pyfoal.CACHE_DIR / 'arctic'
    cache_directory.mkdir(parents=True, exist_ok=True)

    # Iterate over speakers and copy
    for index, speaker in enumerate(ARCTIC_SPEAKERS):
        input_directory = data_directory / f'cmu_us_{speaker}_arctic'

        # Setup output directory
        output_directory = cache_directory / f'{index:04d}'
        output_directory.mkdir(exist_ok=True, parents=True)

        # Get audio files
        audio_files = sorted((input_directory / 'wav').rglob('*.wav'))

        # Get alignment files
        # Note that one wav file is missing
        alignment_files = [
            input_directory / 'lab' / f'{file.stem}.lab'
            for file in audio_files]

        # Load text
        text_file = input_directory / 'etc' / 'txt.done.data'
        with open(text_file) as file:
            lines = [line.strip()[2:-2] for line in file.readlines()]
            text = {line[:12]: line[14:-1] for line in lines}

        # Save to cache
        i = 0
        iterator = pyfoal.iterator(
            zip(alignment_files, audio_files),
            'Formatting arctic',
            total=len(alignment_files))
        for alignment_file, audio_file in iterator:
            assert alignment_file.stem == audio_file.stem

            # Skip files without text
            try:
                sentence = text[audio_file.stem]
            except Exception as error:
                continue

            # Save text
            with open(output_directory / f'{i:06d}.txt', 'w') as file:
                file.write(sentence)

            # Resample audio
            audio = pyfoal.load.audio(audio_file)

            # Save audio
            torchaudio.save(
                output_directory / f'{i:06d}.wav',
                audio,
                pyfoal.SAMPLE_RATE)

            # Load alignment
            with open(alignment_file) as file:

                # Get valid lines
                lines = [line for line in file.readlines()]
                lines = lines[lines.index('#\n') + 1:]
                lines = [line.split() for line in lines if len(line) > 4]

            # Parse into end times and phonemes
            endtimes, _, phonemes = zip(*lines)

            # Handle silence tokens
            phonemes = [
                '<silent>' if phoneme in ['pau', 'ssil'] else phoneme.upper()
                for phoneme in phonemes]

            # Handle breaths
            phonemes = [
                'HH' if phoneme == 'H#' else phoneme
                for phoneme in phonemes]

            # Convert AX phoneme to be in-domain of grapheme-to-phoneme system
            phonemes = [
                'AH' if phoneme == 'AX' else phoneme for phoneme in phonemes]

            # Handle out-of-domain tokens
            phonemes = [
                phoneme if phoneme in pyfoal.load.phonemes() else '<unk>'
                for phoneme in phonemes]

            # Correct end time
            endtimes = [float(endtime) for endtime in endtimes]
            endtimes[-1] = pyfoal.convert.samples_to_seconds(audio.shape[-1])

            # We don't have word alignments, so we just treat each
            # phoneme as a word
            start = 0
            alignment = []
            for end, phoneme in zip(endtimes, phonemes):
                alignment.append(
                    pypar.Word(phoneme, [pypar.Phoneme(phoneme, start, end)]))
                start = end

            # Write alignment
            pypar.Alignment(alignment).save(
                output_directory / f'{i:06d}.TextGrid')

            i += 1


def libritts():
    """Download libritts dataset"""
    # Create directory for downloads
    source_directory = pyfoal.SOURCES_DIR / 'libritts'
    source_directory.mkdir(exist_ok=True, parents=True)

    # Create directory for unpacking
    data_directory = pyfoal.DATA_DIR / 'libritts'
    data_directory.mkdir(exist_ok=True, parents=True)

    # Download and unpack
    for partition in [
        'train-clean-100',
        'train-clean-360',
        'dev-clean',
        'test-clean']:

        # Download
        url = f'https://us.openslr.org/resources/60/{partition}.tar.gz'
        file = source_directory / f'libritts-{partition}.tar.gz'
        download_file(url, file)

        # Unpack
        with tarfile.open(file, 'r:gz') as tfile:
            tfile.extractall(pyfoal.DATA_DIR)

    # Uncapitalize directory name
    shutil.rmtree(str(data_directory), ignore_errors=True)
    shutil.move(
        str(pyfoal.DATA_DIR / 'LibriTTS'),
        str(data_directory),
        copy_function=shutil.copytree)

    # File locations
    audio_files = sorted(data_directory.rglob('*.wav'))
    text_files = [
        file.with_suffix('.normalized.txt') for file in audio_files]

    # Write audio to cache
    speaker_count = {}
    cache_directory = pyfoal.CACHE_DIR / 'libritts'
    cache_directory.mkdir(exist_ok=True, parents=True)
    with pyfoal.chdir(cache_directory):

        # Iterate over files
        iterator = pyfoal.iterator(
            zip(audio_files, text_files),
            'Formatting libritts',
            total=len(audio_files))
        for audio_file, text_file in iterator:

            # Get file metadata
            speaker, *_ = [
                int(part) for part in audio_file.stem.split('_')]

            # Get current speaker count entry
            if speaker not in speaker_count:
                speaker_count[speaker] = [len(speaker_count), 0]
            index, count = speaker_count[speaker]

            # Load audio at system sample rate
            audio = pyfoal.load.audio(audio_file)

            # Save
            stem = f'{index:04d}/{count:06d}'
            output_file = Path(f'{stem}.wav')
            output_file.parent.mkdir(exist_ok=True, parents=True)
            torchaudio.save(
                output_file.parent / f'{output_file.stem}.wav',
                audio,
                pyfoal.SAMPLE_RATE)
            shutil.copyfile(text_file, output_file.with_suffix('.txt'))

            # Update speaker count
            speaker_count[speaker][1] += 1


###############################################################################
# Utilities
###############################################################################


def download_file(url, path):
    """Download file from url"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with open(path, 'wb') as fstream:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)


def download_tar_bz2(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|bz2') as tstream:
            tstream.extractall(path)
