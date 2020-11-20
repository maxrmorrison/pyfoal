import os

import pyfoal


###############################################################################
# Tests
###############################################################################


def test_align():
    """Test forced alignment"""
    alignment = pyfoal.from_file(path('test.wav'), path('test.txt'))
    # TODO - assertions
    import pdb; pdb.set_trace()


###############################################################################
# Utilities
###############################################################################


def path(file):
    """Load test assets"""
    return os.path.join(os.path.dirname(__file__), 'assets', file)
