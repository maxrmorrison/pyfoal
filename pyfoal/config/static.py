import pyfoal


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = pyfoal.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = pyfoal.ASSETS_DIR / 'checkpoints'

# Default configuration file
DEFAULT_CONFIGURATION = pyfoal.ASSETS_DIR / 'configs' / 'radtts.py'


###############################################################################
# Phoneme sets
###############################################################################


# Determine which phoneme set to use
if pyfoal.G2P == 'cmu':
    PHONEMES = pyfoal.CMU_PHONEMES
elif pyfoal.G2P == 'ipa':
    PHONEMES = pyfoal.IPA_PHONEMES
else:
    raise ValueError(f'Grapheme-to-phoneme method {pyfoal.G2P} is not defined')
