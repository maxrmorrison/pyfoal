# Python forced alignment

[![PyPI](https://img.shields.io/pypi/v/pypar.svg)](https://pypi.python.org/pypi/pyfoal)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- [![Downloads](https://pepy.tech/badge/torchcrepe)](https://pepy.tech/project/pypar) -->

This is a modified implementation of the Penn Phonetic Forced Aligner (P2FA)
[1]. Relative to the original implementation, this repo provides the following.
 - Support for Python 3
 - Support for performing forced alignment both in Python and on the
   command-line
 - Fewer alignment failures due to, e.g., out-of-vocabulary (OOV) words or
   punctuation
 - Direct integration with [`pypar`](https://github.com/maxrmorrison/pypar),
   a feature-rich phoneme alignment representation.
 - Multiprocessing for quickly aligning speech datasets
 - Clean, documented code


## Installation

### Hidden Markov Model Toolkit (HTK)
`pyfoal` depends on [HTK](http://htk.eng.cam.ac.uk/) and has been
tested on Mac OS and Linux using HTK version 3.4.0. There are known issues in
using version 3.4.1 on Linux. HTK is released under a license that prohibits
redistribution, so you must install HTK yourself and verify that the commands
`HCopy` and `HVite` are available as system-wide binaries. After downloading
HTK, I use the following for installation on Linux.

```
sudo apt-get install -y gcc-multilib libx11-dev
./configure --disable-hslab
make all
sudo make install
```

For more help with HTK installation, see notes by
[Jaekoo Kang](https://github.com/jaekookang/p2fa_py3#install-htk) and
[Steve Rubin](https://github.com/ucbvislab/p2fa-vislab#install-htk-34-note-341-will-not-work-get-htk-here).


### Python dependencies

Clone this repo and run `pip install -e pyfoal/`.


## Usage


### Force-align text and audio

```python
alignment = pyfoal.align(text, audio, sample_rate)
```

`text` is a string containing the speech transcript.
`audio` is a 1D numpy array containing the speech audio.


### Force-align from files

```python
# Return the resulting alignment
alignment = pyfoal.from_file(text_file, audio_file)

# Save alignment to json
pyfoal.from_file_to_file(text_file, audio_file, output_file)
```

If you need to align many files, use `from_files_to_files`, which accepts
lists of files and uses multiprocessing.


### Command-line interface

```
usage: python -m pyfoal
    [-h]
    --text TEXT [TEXT ...]
    --audio AUDIO [AUDIO ...]
    --output OUTPUT [OUTPUT ...]

optional arguments:
  -h, --help            show this help message and exit
  --text TEXT [TEXT ...]
                        The speech transcript files
  --audio AUDIO [AUDIO ...]
                        The speech audio files
  --output OUTPUT [OUTPUT ...]
                        The json files to save the alignments
```


## Tests

Tests can be run as follows.

```
pip install pytest
pytest
```


## References
[1] J. Yuan and M. Liberman, “Speaker identification on the scotus
corpus,” Journal of the Acoustical Society of America, vol. 123, p.
3878, 2008.
