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

    # Create loader
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=pyfoal.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=pyfoal.data.collate,
        batch_sampler=sampler)
