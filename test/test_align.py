import importlib
import tempfile
import warnings
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

        # Align using p2fa
        with pyfoal.backend('p2fa'):
            pyfoal.from_files_to_files(
                [path('test.txt')],
                [path('test.wav')],
                [Path(directory) / 'p2fa.json'])
            _ = pypar.Alignment(Path(directory) / 'p2fa.json')

        # Maybe align using mfa
        if importlib.util.find_spec('montreal_forced_aligner') is not None:
            pyfoal.from_files_to_files(
                [path('test.txt')],
                [path('test.wav')],
                [Path(directory) / 'mfa.json'])
            _ = pypar.Alignment(Path(directory) / 'mfa.json')
        else:
            warnings.warn('Could not load montreal forced aligner backend')


###############################################################################
# Utilities
###############################################################################


def path(file):
    """Test asset path name resolution"""
    return Path(__file__).parent / 'assets' / file
