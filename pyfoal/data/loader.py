import torch

import pyfoal


def loader(datasets, partition, gpu=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=pyfoal.data.Dataset(datasets, partition),
        batch_size=pyfoal.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=pyfoal.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=pyfoal.data.collate)
