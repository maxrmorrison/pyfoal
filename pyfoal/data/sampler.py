import math

import torch

import pyfoal


###############################################################################
# Sampler selection
###############################################################################


def sampler(dataset, partition):
    """Create batch sampler"""
    # Maybe use distributed sampler for training
    if partition == 'train':
        if torch.distributed.is_initialized():
            return DistributedSampler(dataset)
        else:
            return Sampler(dataset)

    # Deterministic random sampler for validation
    elif partition == 'valid':
        return Sampler(dataset)

    # Sample test data sequentially
    elif partition == 'test':
        return torch.utils.data.SequentialSampler(dataset)

    else:
        raise ValueError(f'Partition {partition} is not defined')


###############################################################################
# Samplers
###############################################################################


class DistributedSampler:

    def __init__(self, dataset):
        super().__init__()
        self.epoch = 0
        self.rank = torch.distributed.get_rank()
        self.num_replicas = torch.distributed.get_world_size()
        self.length = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.length * self.num_replicas
        self.buckets = dataset.buckets()

    def __iter__(self):
        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(pyfoal.RANDOM_SEED + self.epoch)

        # Shuffle buckets
        buckets = [
            self.buckets[i] for i in
            torch.randperm(len(self.buckets), generator=generator).tolist()]

        # Get shuffled indices from shuffled buckets
        indices = []
        for bucket in buckets:
            indices.extend([
                bucket[i] for i in
                torch.randperm(len(bucket), generator=generator).tolist()])

        # Add extra samples to make it evenly divisible
        padding = self.total_size - len(indices)
        if padding <= len(indices):
            indices += indices[:padding]
        else:
            indices += (
                indices * math.ceil(padding / len(indices)))[:padding]

        # Subsample
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch


class Sampler:

    def __init__(self, dataset):
        self.epoch = 0
        self.length = len(dataset)
        self.buckets = dataset.buckets()

    def __iter__(self):
        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(pyfoal.RANDOM_SEED + self.epoch)

        # Shuffle buckets
        buckets = [
            self.buckets[i] for i in
            torch.randperm(len(self.buckets), generator=generator).tolist()]

        # Get shuffled indices from shuffled buckets
        indices = []
        for bucket in buckets:
            indices.extend([
                bucket[i] for i in
                torch.randperm(len(bucket), generator=generator).tolist()])

        # Iterate over shuffled indices
        for index in indices:
            yield index

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
