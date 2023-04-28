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
# Evaluation
###############################################################################


# Timer for benchmarking generation
TIMER = pyfoal.time.Context()
