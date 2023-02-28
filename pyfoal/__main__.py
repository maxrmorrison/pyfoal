import argparse
from pathlib import Path

import pyfoal


###############################################################################
# Forced alignment interface
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text_files',
        nargs='+',
        required=True,
        type=Path,
        help='The speech transcript files')
    parser.add_argument(
        '--audio_files',
        nargs='+',
        required=True,
        type=Path,
        help='The speech audio files')
    parser.add_argument(
        '--output_files',
        nargs='+',
        required=True,
        type=Path,
        help='The files to save the alignments')
    parser.add_argument(
        '--aligner',
        default=pyfoal.ALIGNER,
        help='The alignment method to use')
    parser.add_argument(
        '--num_workers',
        type=int,
        help='Number of CPU cores to utilize. Defaults to all cores.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=pyfoal.DEFAULT_CHECKPOINT,
        help='The checkpoint to use for neural methods')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    return parser.parse_args()


if __name__ == '__main__':
    pyfoal.from_files_to_files(**vars(parse_args()))
