import tempfile
from pathlib import Path

import pypar

import pyfoal


###############################################################################
# Tests
###############################################################################


def test_align():
    """Test forced alignment"""
    # Align to file in temporary directory
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / 'test.json'

        # Align
        pyfoal.from_files_to_files([path('test.txt')],
                                   [path('test.wav')],
                                   [file])

        # Load alignment
        alignment = pypar.Alignment(file)

    # Error check alignment
    assert len(alignment) == 18
    assert len(alignment.phonemes()) == 54
    assert alignment.start() == 0.
    assert alignment.end() == 5.429931972789116
    assert alignment.duration() == 5.429931972789116


###############################################################################
# Utilities
###############################################################################


def path(file):
    """Test asset path name resolution"""
    return Path(__file__).parent / 'assets' / file
