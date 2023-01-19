import argparse

import pyfoal


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        default=pyfoal.DATASETS,
        nargs='+',
        help='The datasets to partition')
    return parser.parse_args()


pyfoal.partition.datasets(**vars(parse_args()))
