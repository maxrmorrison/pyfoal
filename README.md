<h1 align="center">Python forced alignment</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/pyfoal.svg)](https://pypi.python.org/pypi/pyfoal)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/pyfoal)](https://pepy.tech/project/pyfoal)

</div>

Forced alignment suite. Includes English grapheme-to-phoneme (G2P) and
phoneme alignment from the following forced alignment tools.
 - RAD-TTS [1]
 - Montreal Forced Aligner (MFA) [2]
 - Penn Phonetic Forced Aligner (P2FA) [3]

RAD-TTS is used by default. Alignments can be saved to disk or accessed via the
`pypar.Alignment` phoneme alignment representation. See
[`pypar`](https://github.com/maxrmorrison/pypar) for more details.

`pyfoal` also includes the following
 - Converting alignments to and from a categorical representation
   suitable for training machine learning models (`pyfoal.convert`)
 - Natural interpolation of forced alignments for time-stretching speech
   (`pyfoal.interpolate`)


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface](#application-programming-interface)
        * [`pyfoal.from_text_and_audio`](#pyfoalfrom_text_and_audio)
        * [`pyfoal.from_file`](#pyfoalfrom_file)
        * [`pyfoal.from_file_to_file`](#pyfoalfrom_file_to_file)
        * [`pyfoal.from_files_to_files`](#pyfoalfrom_files_to_files)
    * [Command-line interface](#command-line-interface)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
- [Evaluation](#evaluation)
    * [Evaluate](#evaluate)
    * [Plot](#plot)
- [References](#references)


## Installation

`pip install pyfoal`

MFA and P2FA both require additional installation steps found below.


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


## Inference

**TODO** - update

### Force-align text and audio

```python
alignment = pyfoal.from_text_and_audio(text, audio, sample_rate, gpu=gpu)
```

`text` is a string containing the speech transcript
`audio` is a torc containing the speech audio
`sample_rate` is the integer sampling rate
`gpu` is the integer index of the GPU to run alignment on (or `None` for CPU)


### Force-align from files

```python
# Return the resulting alignment
alignment = pyfoal.from_file(text_file, audio_file)

# Save alignment to .json, .mlf, or .TextGrid
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

**TODO** - update

```
usage: python -m pyfoal
    [-h]
    --text TEXT [TEXT ...]
    --audio AUDIO [AUDIO ...]
    --output OUTPUT [OUTPUT ...]
    [--num_workers NUM_WORKERS]

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
```


## Training

### Download

`python -m pyfoal.data.download`

Downloads and uncompresses the `arctic` and `libritts` datasets used for training.


### Preprocess

`python -m pyfoal.data.preprocess`

Converts each dataset to a common format on disk ready for training.


### Partition

`python -m pyfoal.partition`

Generates `train` `valid`, and `test` partitions for `arctic` and `libritts`.
Partitioning is deterministic given the same random seed. You do not need to
run this step, as the original partitions are saved in
`pyfoal/assets/partitions`.


### Train

`python -m pyfoal.train --config <config> --gpus <gpus>`

Trains a model according to a given configuration on the `libritts`
dataset. Uses a list of GPU indices as an argument, and uses distributed
data parallelism (DDP) if more than one index is given. For example,
`--gpus 0 3` will train using DDP on GPUs `0` and `3`.


### Monitor

Run `tensorboard --logdir runs/`. If you are running training remotely, you
must create a SSH connection with port forwarding to view Tensorboard.
This can be done with `ssh -L 6006:localhost:6006 <user>@<server-ip-address>`.
Then, open `localhost:6006` in your browser.


## Evaluation

### Evaluate

```
python -m pyfal.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>`
is the GPU index.

### Plot

**TODO** - alignment plot


## References

[1] R. Badlani, A. Łańcucki, K. J. Shih, R. Valle, W. Ping, and B.
Catanzaro, "One TTS Alignment to Rule Them All," International
Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

[2] J. Yuan and M. Liberman, “Speaker identification on the scotus
corpus,” Journal of the Acoustical Society of America, vol. 123, p.
3878, 2008.

[3] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger,
"Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi,"
Interspeech, vol. 2017, p. 498-502. 2017.
