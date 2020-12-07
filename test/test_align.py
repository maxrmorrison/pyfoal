import os

import pyfoal


###############################################################################
# Tests
###############################################################################


def test_align():
    """Test forced alignment"""
    alignment = pyfoal.from_file(path('test.wav'), path('test.txt'))
    assert len(alignment) == 19
    assert len(alignment.phonemes()) == 55
    assert alignment.start() == 0.
    assert alignment.end() == 5.41
    assert alignment.duration() == 5.41


###############################################################################
# Utilities
###############################################################################


def path(file):
    """Load test assets"""
    return os.path.join(os.path.dirname(__file__), 'assets', file)
