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
        return torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            1,
            False)

    else:
        raise ValueError(f'Partition {partition} is not defined')


###############################################################################
# Samplers
###############################################################################


class Sampler:

    def __init__(self, dataset):
        self.epoch = 0
        self.length = len(dataset)
        self.buckets = dataset.buckets()

    def __iter__(self):
        return iter(self.batch())

    def __len__(self):
        return self.length

    def batch(self):
        """Produces batch indices for one epoch"""
        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(pyfoal.RANDOM_SEED + self.epoch)

        # Make variable-length batches with roughly equal number of frames
        batches = []
        for max_length, bucket in self.buckets:

            # Shuffle bucket
            bucket = bucket[
                torch.randperm(len(bucket), generator=generator).tolist()]

            # Get current batch size
            size = pyfoal.MAX_FRAMES // max_length
            
            # Make batches
            batches.extend(
                [bucket[i:i + size] for i in range(0, len(bucket), size)])

        # Shuffle
        return [
            batches[i] for i in 
            torch.randperm(len(batches), generator=generator).tolist()]

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(Sampler):

    def __init__(self, dataset):
        super().__init__()
        self.rank = torch.distributed.get_rank()
        self.num_replicas = torch.distributed.get_world_size()
        self.length = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.length * self.num_replicas

    def __iter__(self):
        # Divide among GPUs
        return iter(self.batch()[self.rank:self.total_size:self.num_replicas])
