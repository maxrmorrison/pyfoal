
import torch

import pyfoal


###############################################################################
# Dataloader
###############################################################################


def loader(dataset, partition=None, gpu=None):
    """Retrieve a data loader"""
    # Get dataset
    dataset = pyfoal.data.Dataset(dataset, partition)

    # Get sampler
    sampler = pyfoal.data.sampler(dataset, partition)

    # Get batch size
    if partition == 'train':

        # Maybe split batch over GPUs
        if torch.distributed.is_initialized():
            batch_size = \
                pyfoal.BATCH_SIZE // torch.distributed.get_world_size()
        else:
            batch_size = pyfoal.BATCH_SIZE

    elif partition == 'valid':
        batch_size = pyfoal.BATCH_SIZE
    elif partition == 'test':
        batch_size = 1
    else:
        raise ValueError(f'Partition {partition} is not defined')

    # Create loader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=pyfoal.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=pyfoal.data.collate,
        sampler=sampler)
