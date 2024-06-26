MODULE = 'pyfoal'

# Configuration name
CONFIG = 'radtts-trf3-mrf3-512-aw128-temp0005'

# Width of the attention layer
ATTENTION_WIDTH = 128

# Maximum number of frames in a batch (per GPU)
MAX_FRAMES = 60000

# Kernel sizes for the mel encoder
MEL_ENCODER_KERNEL_SIZES = [3, 1, 1]

# Width of the phoneme embedding
PHONEME_EMBEDDING_SIZE = 512

# Sampling temperature
TEMPERATURE = .0005

# Kernel sizes for the text encoder
TEXT_ENCODER_KERNEL_SIZES = [3, 1, 1, 1, 1]
