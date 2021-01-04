import argparse
from pathlib import Path

import pyfoal


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        '--text',
        nargs='+',
        required=True,
        type=Path,
        help='The speech transcript files')
    parser.add_argument(
        '--audio',
        nargs='+',
        required=True,
        type=Path,
        help='The speech audio files')
    parser.add_argument(
        '--output',
        nargs='+',
        required=True,
        type=Path,
        help='The json files to save the alignments')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Generate alignment and save to disk
    pyfoal.from_files_to_files(args.text, args.audio, args.output)
