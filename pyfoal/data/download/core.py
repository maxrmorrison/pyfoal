import json
import re
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

    # Download text data
    download_file(
        'http://festvox.org/cmu_arctic/cmuarctic.data',
        data_directory / 'sentences.txt')

    # Setup data directory
    cache_directory = pyfoal.CACHE_DIR / 'arctic'
    cache_directory.mkdir(parents=True, exist_ok=True)

    # Load text
    text_file = data_directory / 'sentences.txt'
    with open(text_file, 'r') as file:
        content = file.read()

    # Write csv with text
    sentences = {
        match[0]: match[1] for match in re.findall(
            r'\( (arctic_[ab][0-9][0-9][0-9][0-9]) \"(.+)\" \)',
            content,
            re.MULTILINE)}

    # Iterate over speakers and copy
    iterator = pyfoal.iterator(
        ARCTIC_SPEAKERS,
        'Formatting arctic',
        total=len(ARCTIC_SPEAKERS))
    for speaker in iterator:
        input_directory = data_directory / f'cmu_us_{speaker}_arctic'
        output_directory = cache_directory / speaker

        # Create output directories
        for feature in ['alignment', 'audio', 'text']:
            (output_directory / feature).mkdir(exist_ok=True, parents=True)

        # Get input files
        alignment_files = sorted((input_directory / 'lab').rglob('*.lab'))
        alignment_files = [
            file for file in alignment_files if file.stem != '*']
        audio_files = sorted((input_directory / 'wav').rglob('*.wav'))

        # Save to cache
        for alignment_file, audio_file in zip(alignment_files, audio_files):
            assert alignment_file.stem == audio_file.stem

            try:

                # Save text
                text_file = output_directory / 'text' / f'{audio_file.stem}.txt'
                with open(text_file, 'w') as file:
                    file.write(sentences[audio_file.stem])

            except KeyError:

                # Skip examples with no available text
                continue

            # Resample audio
            audio = pyfoal.load.audio(audio_file)

            # Save audio
            torchaudio.save(
                output_directory / 'audio' / audio_file.name,
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
                'sp' if phoneme == 'pau' else phoneme
                for phoneme in phonemes]

            # Handle out-of-domain tokens
            phonemes = [
                phoneme if phoneme in pyfoal.PHONEMES else '<unk>'
                for phoneme in phonemes]

            # Correct end time
            endtimes = [float(endtime) for endtime in endtimes]
            endtimes[-1] = pyfoal.convert.samples_to_seconds(audio.shape[-1])

            # Handle duplicates
            i = 0
            while i < len(phonemes) - 1:
                if phonemes[i] == phonemes[i + 1]:
                    endtimes[i] = endtimes[i + 1]
                    del phonemes[i]
                else:
                    i += 1

            # We don't have word alignments, so we just treat each
            # phoneme as a word
            start = 0
            alignment = []
            for end, phoneme in zip(endtimes, phonemes):
                alignment.append(
                    pypar.Word(phoneme, [pypar.Phoneme(phoneme, start, end)]))
                end = start

            # Write alignment
            pypar.Alignment(alignment).save(
                output_directory / 'alignment' / f'{audio_file.stem}.TextGrid')


def libritts():
    """Download libritts dataset"""
    # Create directory for downloads
    source_directory = pyfoal.SOURCE_DIR / 'libritts'
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

    # Track previous and next utterances
    context = {}
    prev_parts = None

    # Write audio to cache
    speaker_count = {}
    cache_directory = pyfoal.CACHE_DIR / 'libritts'
    cache_directory.mkdir(exist_ok=True, parents=True)
    with pyfoal.data.chdir(cache_directory):

        # Iterate over files
        iterator = pyfoal.iterator(
            zip(audio_files, text_files),
            'Formatting libritts',
            total=len(audio_files))
        for audio_file, text_file in iterator:

            # Get file metadata
            speaker, book, chapter, utterance = [
                int(part) for part in audio_file.stem.split('_')]

            # Update speaker and get current entry
            if speaker not in speaker_count:

                # Each entry is (index, count)
                speaker_count[speaker] = [len(speaker_count), 0]

            speaker_count[speaker][1] += 1
            index, count = speaker_count[speaker]

            # Load audio
            audio, sample_rate = torchaudio.load(audio_file)

            # Save at system sample rate
            stem = f'{index:04d}/{count:06d}'
            output_file = Path(f'{stem}.wav')
            output_file.parent.mkdir(exist_ok=True, parents=True)
            audio = pyfoal.resample(audio, sample_rate)
            torchaudio.save(
                output_file.parent / f'{output_file.stem}.wav',
                audio,
                pyfoal.SAMPLE_RATE)
            shutil.copyfile(text_file, output_file.with_suffix('.txt'))

            # Update context
            if (
                prev_parts is not None and
                prev_parts == (speaker, book, chapter, utterance - 1)
            ):
                prev_stem = f'{index:04d}/{count - 1:06d}'
                context[stem] = { 'prev': prev_stem, 'next': None }
                context[prev_stem]['next']: stem
            else:
                context[stem] = { 'prev': None, 'next': None }
            prev_parts = (speaker, book, chapter, utterance)

        # Save context information
        with open('context.json', 'w') as file:
            json.dump(context, file, indent=4, sort_keys=True)


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
