import argparse

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
        help='The names of the datasets to preprocess')
    return parser.parse_args()


pyfoal.data.preprocess.datasets(**vars(parse_args()))
