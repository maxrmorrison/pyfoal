from pathlib import Path


###############################################################################
# Constants
###############################################################################


# The aligner to use. One of ['p2fa', 'mfa'].
ALIGNER = 'mfa'

# The location of the aligner model and phoneme dictionary
ASSETS_DIR = Path(__file__).parent / 'assets'

# The default audio sampling rate of P2FA
P2FA_SAMPLE_RATE = 11025
