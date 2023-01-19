import json
import random

import pyfoal


def datasets(datasets=pyfoal.DATASETS):
    """Partition datasets"""
    for dataset in datasets:

        # Random seed
        random.seed(pyfoal.RANDOM_SEED)

        # TODO - make partition dictionary
        partition = {'train': [], 'valid': [], 'test': []}

        # Save to disk
        file = pyfoal.PARTITION_DIR / f'{dataset}.json'
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, indent=4)
