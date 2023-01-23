from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'pyfoal'


###############################################################################
# Audio parameters
###############################################################################


# The audio sampling rate
SAMPLE_RATE = 16000  # samples per second


###############################################################################
# Data parameters
###############################################################################


# Names of all datasets
DATASETS = ['arctic', 'libritts']

# Datasets for evaluation
EVALUATION_DATASETS = ['arctic']


###############################################################################
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'

# Location of compressed datasets on disk
SOURCES_DIR = Path(__file__).parent.parent.parent / 'data' / 'sources'


###############################################################################
# Evaluation parameters
###############################################################################


# The method to use for forced alignment. One of ['mfa', 'p2fa', 'radtts'].
ALIGNER = 'radtts'

# The method to use for grapheme-to-phoneme conversion. One of ['cmu', 'ipa'].
G2P = 'ipa'

# Number of steps between tensorboard logging
LOG_INTERVAL = 2500  # steps

# Number of steps to perform for tensorboard logging
LOG_STEPS = 16


###############################################################################
# Training parameters
###############################################################################


# Batch size (per gpu)
BATCH_SIZE = 64

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of training steps
STEPS = 300000

# Number of data loading worker threads
NUM_WORKERS = 2

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# Model parameters
###############################################################################


# Sampling temperature
TEMPERATURE = .0005


###############################################################################
# Phoneme sets
###############################################################################


# CMU phoneme set
CMU_PHONEMES = [
	'aa',
	'ae',
	'ah',
	'ao',
	'aw',
	'ay',
	'b',
	'ch',
	'd',
	'dh',
	'eh',
	'er',
	'ey',
	'f',
	'g',
	'hh',
	'ih',
	'iy',
	'jh',
	'k',
	'l',
	'm',
	'n',
	'ng',
	'ow',
	'oy',
	'p',
	'r',
	's',
	'sh',
	't',
	'th',
	'uh',
	'uw',
	'v',
	'w',
	'y',
	'z',
	'zh',
	'ax',
	'sp',
	'<unk>'
]

# IPA phoneme set
# TODO
IPA_PHONEMES = []
