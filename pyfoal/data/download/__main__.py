import argparse

import pyfoal


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=pyfoal.DATASETS,
        help='The datasets to download')
    return parser.parse_args()


pyfoal.data.download.datasets(**vars(parse_args()))
