import argparse

import pyfoal


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        '--audio',
        nargs='+',
        help='The audio files to process')
    parser.add_argument(
        '--text',
        nargs='+',
        help='The corresponding transcript files')
    parser.add_argument(
        '--output',
        nargs='+',
        help='The files to save the alignments')

    # Optional arguments
    parser.add_argument('--tmpdir', help='Directory to store temporary files')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Generate alignment and save to disk
    pyfoal.from_files_to_files(args.audio,
                               args.text,
                               args.output_file,
                               args.tmpdir)
