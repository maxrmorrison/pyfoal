###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('pyfoal', defaults)

# Import configuration parameters
from .config.defaults import *
from . import time
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .core import *
from .interpolate import is_voiced, is_vowel
from .model import Model
from . import baselines
from . import checkpoint
from . import convert
from . import data
from . import evaluate
from . import g2p
from . import interpolate
from . import load
from . import loudness
from . import model
from . import partition
from . import plot
from . import train
from . import viterbi
from . import write
