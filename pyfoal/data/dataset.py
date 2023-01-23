import pypar
import torch

import pyfoal


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset"""

    def __init__(self, datasets, partition):
        self.partition = partition
        self.datasets = {
            dataset: Metadata(dataset, partition) for dataset in datasets}

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        # Get dataset and index into dataset
        index, dataset = self.get_dataset(index)

        # Get dataset stem
        stem = dataset.stems[index]

        # Load phoneme indices
        text = torch.load(dataset.cache / 'text' / f'{stem}.pt')

        # Load audio
        audio = pyfoal.load.audio(dataset.cache / 'audio' / f'{stem}.wav')

        # Maybe load true alignment
        if dataset.name == 'arctic':
            alignment = pypar.Alignment(
                dataset.cache / 'alignment' / f'{stem}.TextGrid')

            # Compute word bounds from alignment
            bounds = alignment.word_bounds(
                pyfoal.SAMPLE_RATE,
                pyfoal.HOPSIZE,
                silences=True)
            bounds = torch.cat(
                [torch.tensor(bound)[None] for bound in bounds]).T
        else:
            bounds = None

        return text, audio, bounds, stem

    def __len__(self):
        """Length of the dataset"""
        return sum(len(dataset) for dataset in self.datasets)

    def get_dataset(self, index):
        """Retrieve the dataset to index and index into the datset"""
        i = 0
        dataset = self.datasets[i]
        upper_bound = len(dataset)
        while index >= upper_bound:
            i += 1
            dataset = self.datasets[i]
            upper_bound += len(dataset)

        # Get index into dataset
        index -= (upper_bound - len(dataset))

        return index, dataset


###############################################################################
# Metadata
###############################################################################


class Metadata:

    def __init__(self, name, partition):
        self.name = name
        self.cache = pyfoal.CACHE_DIR / name
        self.stems = pyfoal.load.partition(name)[partition]

    def __len__(self):
        return len(self.stems)
