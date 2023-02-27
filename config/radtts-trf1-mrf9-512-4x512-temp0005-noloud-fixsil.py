MODULE = 'pyfoal'

# Configuration name
CONFIG = 'radtts-trf1-mrf9-512-4x512-temp0005-noloud-fixsil'

# Whether to allow silent tokens on loud frames
ALLOW_LOUD_SILENCE = False

# Kernel sizes for the mel encoder
MEL_ENCODER_KERNEL_SIZES = [3, 3, 3, 3]

# Mel layer widths
MEL_ENCODER_WIDTHS = [512, 512, 512]

# Width of the phoneme embedding
PHONEME_EMBEDDING_SIZE = 512

# Sampling temperature
TEMPERATURE = .0005

# Kernel sizes for the text encoder
TEXT_ENCODER_KERNEL_SIZES = [1, 1, 1, 1, 1]
