###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('pyfoal', defaults)

# Import configuration parameters
from .config.static import *
from .config.defaults import *


###############################################################################
# Module imports
###############################################################################

from .core import *
from .interpolate import is_voiced, is_vowel
from . import baselines
from . import convert
from . import data
from . import evaluate
from . import interpolate
from . import load
from . import model
from . import partition
from . import train
