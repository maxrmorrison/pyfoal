MODULE = 'pyfoal'

# Configuration name
CONFIG = 'radtts-trf3-mrf3-512-4x512-temp0005'

# Kernel sizes for the mel encoder
MEL_ENCODER_KERNEL_SIZES = [3, 1, 1, 1]

# Mel layer widths
MEL_ENCODER_WIDTHS = [512, 512, 512]

# Width of the phoneme embedding
PHONEME_EMBEDDING_SIZE = 512

# Sampling temperature
TEMPERATURE = .0005

# Kernel sizes for the text encoder
TEXT_ENCODER_KERNEL_SIZES = [3, 1, 1, 1, 1]
