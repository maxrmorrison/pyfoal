import csv
import re
import requests
import shutil
import tarfile

import tqdm

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
    # Setup source directory
    source_directory = pyfoal.SOURCES_DIR / 'arctic'
    source_directory.mkdir(parents=True, exist_ok=True)

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
        download_tar_bz2(url.format(speaker), source_directory)

    # Download text data
    download_file(
        'http://festvox.org/cmu_arctic/cmuarctic.data',
        source_directory / 'sentences.txt')

    # Setup data directory
    data_directory = pyfoal.DATA_DIR / 'arctic'
    data_directory.mkdir(parents=True, exist_ok=True)

    # Load text
    text_file = source_directory / 'sentences.txt'
    with open(text_file, 'r') as file:
        content = file.read()

    import pdb; pdb.set_trace()

    # Write csv with text
    # TODO - corresponding .txt files instead
    new_sentences_file = data_directory / 'sentences.csv'
    rows = [
        match for match in re.findall(
            r'\( (arctic_[ab][0-9][0-9][0-9][0-9]) \"(.+)\" \)',
            content,
            re.MULTILINE)]
    with open(new_sentences_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'prompt'])
        writer.writerows(rows)

    # Get speaker directories
    speakers = [
        source_directory / f'cmu_us_{speaker}_arctic'
        for speaker in ARCTIC_SPEAKERS]

    # Iterate over speaker directories and copy
    iterator = pyfoal.iterator(
        speakers,
        'Formatting arctic speakers',
        total=len(speakers))
    for speaker in iterator:

        # Get file name map based in arctic version 0.90 or 0.95
        if speaker.name == 'cmu_us_awb_arctic':
            v90 = speaker / 'etc' / 'txt.done.data'
            v95 = text_file
            with open(v90) as f:
                cv90 = f.read()
            with open(v95) as f:
                cv95 = f.read()
            def id_map(id): return v0_90_to_v0_95(id, cv90, cv95)
        else:
            def id_map(id): return id
        new_speaker_dir = data_directory / speaker.name

        # Get label files
        lab_dir_path = speaker / 'lab'
        lab_files = files_with_extension('lab', lab_dir_path)

        # Create destination directory
        new_lab_dir_path = new_speaker_dir / 'lab'
        new_lab_dir_path.mkdir(parents=True, exist_ok=True)

        # Transfer phoneme label files
        wav_dir_path = speaker / 'wav'
        new_phone_files = []
        nested_iterator = tqdm.tqdm(
            lab_files,
            desc=f'transferring phonetic label files for arctic speaker {speaker.name}',
            total=len(lab_files),
            dynamic_ncols=True)
        for lab_file in nested_iterator:

            # Handle extra file included in some arctic versions
            if lab_file.stem == '*':
                continue

            with open(lab_file, 'r') as f:
                lines = f.readlines()

                # Remove header
                non_header_lines = lines[lines.index('#\n')+1:]

                # Extract phonemes and timestamps
                timestamps, _, phonemes = zip(
                    *[line.split() for line in non_header_lines if len(line) >= 5])

                # Handle various silence tokens
                phonemes = [
                    'sp' if phone == 'pau' else phone for phone in phonemes]

                # Handle out-of-domain tokens
                # TODO - get phoneme list
                phonemes = [
                    phone if phone in ppgs.PHONEME_LIST else '<unk>'
                    for phone in phonemes]

            # Correct final duration of timestamp
            # TODO - do this without loading
            with open(wav_dir_path / (lab_file.stem + '.wav'), 'rb') as f:
                audio = pyfoal.load.audio(f)
                audio_duration = audio[0].shape[0] / pyfoal.SAMPLE_RATE
                if not abs(audio_duration - float(timestamps[-1])) <= 1e-1:
                    print(f'failed with stem {lab_file.stem}')
                    continue
                timestamps = list(timestamps)
                timestamps[-1] = str(audio_duration)

            rows = zip(timestamps, phonemes)
            # write new label file as CSV
            try:
                new_phone_file = new_lab_dir_path / \
                    (id_map(lab_file.stem) + '.csv')
            except TypeError:
                continue
            new_phone_files.append(new_phone_file)
            with open(new_phone_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'phoneme'])
                writer.writerows(rows)

        # Copy audio files
        new_wav_dir_path = new_speaker_dir / 'wav'
        new_wav_dir_path.mkdir(parents=True, exist_ok=True)
        wav_files = files_with_extension('wav', wav_dir_path)
        for wav_file in wav_files:
            shutil.copy(wav_file, new_wav_dir_path / (id_map(wav_file.stem) + '.wav'))


def libritts():
    """Download libritts dataset"""
    # TODO
    pass


###############################################################################
# Utilities
###############################################################################


def ci_fmt(fragment):
    """Create case insensitive glob fragment"""
    characters = list(fragment.lower())
    return ''.join([f'[{c}{c.upper()}]' for c in characters])


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
        with tarfile.open(fileobj=rstream.raw, mode="r|bz2") as tstream:
            tstream.extractall(path)


def files_with_extension(ext, path):
    return list(path.rglob(f"*.{ci_fmt(ext)}"))


def v0_90_to_v0_95(id, v90_sentences, v95_sentences):
    """maps cmu_arctic data ids from version 0.90 to version 0.95 by taking an id and the
    contents of a prompt file for v90 and a prompt_file for v95"""
    sentence = re.search(rf'\( {id} \"(.+)\" \)', v90_sentences).groups()[0]
    try:
        new_id = re.search(
            rf'\( (arctic_[ab][0-9][0-9][0-9][0-9]) \"{sentence}\" \)', v95_sentences).groups()[0]
    except AttributeError:
        return None
    return new_id
