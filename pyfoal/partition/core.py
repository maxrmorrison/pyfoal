import json
import random

import pyfoal


###############################################################################
# Partition
###############################################################################


def datasets(datasets=pyfoal.DATASETS):
    """Partition datasets"""
    for dataset in datasets:

        # Random seed
        random.seed(pyfoal.RANDOM_SEED)

        # Partition
        if dataset == 'arctic':
            partition = arctic()
        elif dataset == 'libritts':
            partition = libritts()
        else:
            raise ValueError(f'Dataset {dataset} is not defined')

        # Save to disk
        file = pyfoal.PARTITION_DIR / f'{dataset}.json'
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, indent=4)


###############################################################################
# Partition individiual datasets
###############################################################################


def arctic():
    """Partition arctic"""
    # Get text files
    files = (pyfoal.CACHE_DIR / 'arctic').rglob('*.txt')

    # Extract speaker and filename as stem
    stems = sorted([f'{file.parent.name}/{file.stem}' for file in files])

    # Shuffle
    random.shuffle(stems)

    # Get split point
    split = int(.9 * len(stems))

    # Partition
    return {
        'train': [],
        'valid': sorted(stems[split:]),
        'test': sorted(stems[:split])}


def libritts():
    """Partition libritts"""
    # Get text files
    files = list((pyfoal.CACHE_DIR / 'libritts').rglob('*.txt'))

    # Get list of speakers
    speakers = list({file.parent.name for file in files})
    random.shuffle(speakers)

    # Extract speaker and filename as stem
    stems = sorted([f'{file.parent.name}/{file.stem}' for file in files])

    # Get split points
    left, right = int(.9 * len(speakers)), int(.95 * len(speakers))

    # Partition speakers
    train_speakers = speakers[:left]
    valid_speakers = speakers[left:right]
    test_speakers = speakers[right:]

    # Partition stems
    return {
        'train': [stem for stem in stems if stem.split('/')[0] in train_speakers],
        'valid': [stem for stem in stems if stem.split('/')[0] in valid_speakers],
        'test': [stem for stem in stems if stem.split('/')[0] in test_speakers]}
