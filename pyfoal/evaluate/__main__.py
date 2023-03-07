import argparse
from pathlib import Path

import pyfoal


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        default=pyfoal.DATASETS,
        nargs='+',
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        default=pyfoal.DEFAULT_CHECKPOINT,
        type=Path,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_known_args()[0]


pyfoal.evaluate.datasets(**vars(parse_args()))
