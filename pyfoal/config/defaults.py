from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'pyfoal'


###############################################################################
# Audio parameters
###############################################################################


# Audio hopsize
HOPSIZE = 160  # samples

# Number of melspectrogram channels
NUM_MELS = 80

# Number of spectrogram channels
NUM_FFT = 1024

# The audio sampling rate
SAMPLE_RATE = 16000  # samples per second

# Analysis window size
WINDOW_SIZE = 1024


###############################################################################
# Data parameters
###############################################################################


# Number of buckets to partition training and validation data into based on
# length to avoid excess padding
BUCKETS = 8

# Names of all datasets
DATASETS = ['arctic', 'libritts']

# Number of text files per batch for espeak grapheme-to-phoneme
ESPEAK_BATCH_SIZE = 1024

# Datasets for evaluation
EVALUATION_DATASETS = ['arctic']


###############################################################################
# Directories
###############################################################################


# Root location for saving outputs
# TEMPORARY
# ROOT_DIR = Path(__file__).parent.parent.parent
ROOT_DIR = Path('/data/max/pyfoal')

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = ROOT_DIR / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = ROOT_DIR / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = ROOT_DIR / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = ROOT_DIR / 'runs'

# Location of compressed datasets on disk
SOURCES_DIR = ROOT_DIR / 'data' / 'sources'


###############################################################################
# Evaluation parameters
###############################################################################


# The method to use for forced alignment. One of ['mfa', 'p2fa', 'radtts'].
ALIGNER = 'radtts'

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
NUM_WORKERS = 16

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# Model parameters
###############################################################################


# Scale factor for the beta-binomial attention prior
ATTENTION_PRIOR_SCALE_FACTOR = .05

# Width of the phoneme embedding
PHONEME_EMBEDDING_SIZE = 512

# Sampling temperature
TEMPERATURE = .0005
