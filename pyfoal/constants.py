from pathlib import Path


###############################################################################
# Constants
###############################################################################


# The aligner to use. One of ['p2fa', 'mfa'].
ALIGNER = 'mfa'

# The location of the aligner model and phoneme dictionary
ASSETS_DIR = Path(__file__).parent / 'assets'
