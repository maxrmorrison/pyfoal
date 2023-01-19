import torch

import NAME


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The name of the data partition
    """

    def __init__(self, names, partition):
        self.partition = partition
        self.datasets = [Metadata(name, partition) for name in names]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        # Get dataset and index into dataset
        index, dataset = self.get_dataset(index)

        # Get dataset stem
        stem = self.stems[index]

        # TODO - Load from stem
        raise NotImplementedError

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
        self.stems = NAME.load.partition(name)[partition]

    def __len__(self):
        return len(self.stems)
