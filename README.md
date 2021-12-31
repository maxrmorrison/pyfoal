# Python forced alignment

[![PyPI](https://img.shields.io/pypi/v/pypar.svg)](https://pypi.python.org/pypi/pyfoal)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/pyfoal)](https://pepy.tech/project/pyfoal)

Forced alignment suite. Includes English grapheme-to-phoneme (G2P) and
phoneme alignment from the following forced alignment tools.
 - Montreal Forced Aligner (MFA) [1]
 - Penn Phonetic Forced Aligner (P2FA) [2]

MFA is used by default. Alignments can be saved to disk or accessed via the
`pypar.Alignment` phoneme alignment representation. See
[`pypar`](https://github.com/maxrmorrison/pypar) for more details.

`pyfoal` also includes the following
 - Converting alignments to and from a categorical representation
   suitable for training machine learning models (`pyfoal.convert`)
 - Natural interpolation of forced alignments for time-stretching speech
   (`pyfoal.interpolate`)


## Installation

First, install the Python dependencies in a new Conda environment

`pip install pyfoal`

Next, perform the necessary installation of either MFA or P2FA.


### Montreal Forced Aligner (MFA)

`conda install -c conda-forge montreal-forced-aligner`


### Penn Phonetic Forced Aligner (P2FA)

P2FA depends on the
[Hidden Markov Model Toolkit (HTK)](http://htk.eng.cam.ac.uk/), which has been
tested on Mac OS and Linux using HTK version 3.4.0. There are known issues in
using version 3.4.1 on Linux. HTK is released under a license that prohibits
redistribution, so you must install HTK yourself and verify that the commands
`HCopy` and `HVite` are available as system-wide binaries. After downloading
HTK, I use the following for installation on Linux.

```
sudo apt-get install -y gcc-multilib libx11-dev
sudo chmod +x configure
./configure --disable-hslab
make all
sudo make install
```

For more help with HTK installation, see notes by
[Jaekoo Kang](https://github.com/jaekookang/p2fa_py3#install-htk) and
[Steve Rubin](https://github.com/ucbvislab/p2fa-vislab#install-htk-34-note-341-will-not-work-get-htk-here).


### Python dependencies

`pip install pyfoal`


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

```python
# Align many files at once
# num_workers is the number of parallel jobs
pyfoal.from_files_to_files(text_files, audio_files, output_files, num_workers)
```


### Changing backend to use P2FA

```python
# Change backend
with pyfoal.backend('p2fa'):

    # Perform alignment
    alignment = pyfoal.align(text, audio, sample_rate)
```


### Command-line interface

```
usage: python -m pyfoal
    [-h]
    --text TEXT [TEXT ...]
    --audio AUDIO [AUDIO ...]
    --output OUTPUT [OUTPUT ...]
    [--num_workers NUM_WORKERS]
    [--backend BACKEND]

optional arguments:
    -h, --help          show this help message and exit
    --text TEXT [TEXT ...]
                        The speech transcript files
    --audio AUDIO [AUDIO ...]
                        The speech audio files
    --output OUTPUT [OUTPUT ...]
                        The json files to save the alignments
    --num_workers NUM_WORKERS
                        Number of CPU cores to utilize. Defaults to all cores.
    --backend BACKEND
                        The aligner to use. One of ['mfa' (default), 'p2fa'].
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

[2] M. McAuliffe, , M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger,
"Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi,"
Interspeech, vol. 2017, p. 498-502. 2017.
