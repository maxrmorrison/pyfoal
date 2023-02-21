import numpy as np
import pypar
import torch
import torchaudio

import pyfoal


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset"""

    def __init__(self, datasets, partition):
        self.partition = partition
        self.datasets = [
            Metadata(dataset, partition) for dataset in datasets]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        # Get dataset and index into dataset
        index, dataset = self.get_dataset(index)

        # Get dataset stem
        stem = dataset.stems[index]

        # Load phoneme indices
        phonemes = torch.load(dataset.cache / f'{stem}-phonemes.pt')

        # Load audio
        audio = pyfoal.load.audio(dataset.cache / f'{stem}.wav')

        # Compute prior
        prior = torch.load(dataset.cache / f'{stem}-prior.pt')

        # Maybe load true alignment
        if dataset.name == 'arctic':
            alignment = pypar.Alignment(dataset.cache / f'{stem}.TextGrid')
        else:
            alignment = None

        # Load text
        text = pyfoal.load.text(dataset.cache / f'{stem}.txt')

        return phonemes, audio, prior, alignment, text, stem

    def __len__(self):
        """Length of the dataset"""
        return sum(len(dataset) for dataset in self.datasets)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // pyfoal.BUCKETS

        # Get indices in order of length
        lengths = []
        for i in range(len(self)):
            index, dataset = self.get_dataset(i)
            lengths.append(dataset.lengths[index])
        indices = np.argsort(lengths)

        # Split into buckets based on length
        buckets = [indices[i:i + size] for i in range(0, len(self), size)]

        # Add max length of each bucket
        buckets = [(lengths[bucket[-1]], bucket) for bucket in buckets]

        return buckets

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

        # Store lengths for bucketing
        audio_files = list([
            self.cache / f'{stem}.wav' for stem in self.stems])
        self.lengths = [
            pyfoal.convert.samples_to_frames(
                torchaudio.info(audio_file).num_frames)
            for audio_file in audio_files]

    def __len__(self):
        return len(self.stems)
