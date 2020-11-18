import argparse

import pyfoal


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('audio', help='The audio file to process')
    parser.add_argument('text', help='The corresponding transcript file')
    parser.add_argument('output_file', help='The file to save the alignment')
    parser.add_argument('htk_directory', help='The path to the HTK binaries')

    # Optional arguments
    parser.add_argument('--tmpdir', help='Directory to store temporary files')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Generate alignment and save to disk
    pyfoal.from_file_to_file(args.audio,
                             args.text,
                             args.output_file,
                             args.htk_directory,
                             args.tmpdir)
