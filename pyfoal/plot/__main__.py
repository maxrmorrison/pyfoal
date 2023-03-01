import argparse
from pathlib import Path

import pyfoal


###############################################################################
# Plot alignments
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Plot alignments')
    parser.add_argument(
        '--audio',
        type=Path,
        required=True,
        help='The audio file')
    parser.add_argument(
        '--alignment',
        type=Path,
        required=True,
        help='The speech alignment file')
    parser.add_argumeng(
        '--target',
        type=Path,
        help='An optional target alignment to compare')
    return parser.parse_args()


if __name__ == '__main__':
    pyfoal.plot.alignments(**vars(parse_args()))
