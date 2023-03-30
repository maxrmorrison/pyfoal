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

# Datasets for evaluation
EVALUATION_DATASETS = ['arctic']


###############################################################################
# Decoder parameters
###############################################################################


# Whether to account for the padding applied to the mels
ADJUST_PADDING = False

# Whether to allow silent tokens on loud frames
ALLOW_LOUD_SILENCE = True

# Whether to allow spaces to be skipped
ALLOW_SKIP_SPACE = True

# Whether to perform local interpolation over time
INTERPOLATE = False

# Threshold below which audio is considered silent
SILENCE_THRESHOLD = -60.  # dB


###############################################################################
# Directories
###############################################################################


# Root location for saving outputs
ROOT_DIR = Path(__file__).parent.parent.parent

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
LOG_INTERVAL = 5000  # steps

# Number of steps to perform for tensorboard logging
LOG_STEPS = 16

# Number of examples to plot to Tensorboard
PLOT_EXAMPLES = 8


###############################################################################
# Training parameters
###############################################################################


# Maximum number of frames in a batch (per GPU)
MAX_FRAMES = 65000

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of training steps
STEPS = 250000

# Number of data loading worker threads
NUM_WORKERS = 8

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# Model parameters
###############################################################################


# Scale factor for the beta-binomial attention prior
ATTENTION_PRIOR_SCALE_FACTOR = .05

# Width of the attention layer
ATTENTION_WIDTH = 80

# Whether to use LSTM layer in text encoder
LSTM = False

# Kernel sizes for the mel encoder
MEL_ENCODER_KERNEL_SIZES = [3, 1, 1]

# Mel layer widths
MEL_ENCODER_WIDTHS = [80, 160]

# Width of the phoneme embedding
PHONEME_EMBEDDING_SIZE = 512

# Weight to apply to the attention prior
PRIOR_WEIGHT = 1.

# Sampling temperature
TEMPERATURE = .0005

# Kernel sizes for the text encoder
TEXT_ENCODER_KERNEL_SIZES = [5, 5, 5, 3, 1]
