<h1 align="center">Python forced alignment</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/pyfoal.svg)](https://pypi.python.org/pypi/pyfoal)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/pyfoal)](https://pepy.tech/project/pyfoal)

</div>

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
    * [Evaluate](#evaluate)
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

### Force-align text and audio

```python
import pyfoal

# Load text
text = pyfoal.load.text(text_file)

# Load and resample audio
audio = pyfoal.load.audio(audio_file)

# Select an aligner. One of ['mfa', 'p2fa', 'radtts' (default)].
aligner = 'radtts'

# For RAD-TTS, select a model checkpoint
checkpoint = pyfoal.DEFAULT_CHECKPOINT

# Select a GPU to run inference on
gpu = 0

alignment = pyfoal.from_text_and_audio(
    text,
    audio,
    pyfoal.SAMPLE_RATE,
    aligner=aligner,
    checkpoint=checkpoint,
    gpu=gpu)
```


### Application programming interface

#### `pyfoal.from_text_and_audio`


```
"""Phoneme-level forced-alignment

Arguments
    text : string
        The speech transcript
    audio : torch.tensor(shape=(1, samples))
        The speech signal to process
    sample_rate : int
        The audio sampling rate

Returns
    alignment : pypar.Alignment
        The forced alignment
"""
```


#### `pyfoal.from_file`

```
"""Phoneme alignment from audio and text files

Arguments
    text_file : Path
        The corresponding transcript file
    audio_file : Path
        The audio file to process
    aligner : str
        The alignment method to use
    checkpoint : Path
        The checkpoint to use for neural methods
    gpu : int
        The index of the gpu to perform alignment on for neural methods

Returns
    alignment : Alignment
        The forced alignment
"""
```


#### `pyfoal.from_file_to_file`

```
"""Perform phoneme alignment from files and save to disk

Arguments
    text_file : Path
        The corresponding transcript file
    audio_file : Path
        The audio file to process
    output_file : Path
        The file to save the alignment
    aligner : str
        The alignment method to use
    checkpoint : Path
        The checkpoint to use for neural methods
    gpu : int
        The index of the gpu to perform alignment on for neural methods
"""
```


#### `pyfoal.from_files_to_files`

```
"""Perform parallel phoneme alignment from many files and save to disk

Arguments
    text_files : list
        The transcript files
    audio_files : list
        The corresponding speech audio files
    output_files : list
        The files to save the alignments
    aligner : str
        The alignment method to use
    num_workers : int
        Number of CPU cores to utilize. Defaults to all cores.
    checkpoint : Path
        The checkpoint to use for neural methods
    gpu : int
        The index of the gpu to perform alignment on for neural methods
"""
```


### Command-line interface

```
python -m pyfoal
    [-h]
    --text_files TEXT_FILES [TEXT_FILES ...]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--aligner ALIGNER]
    [--num_workers NUM_WORKERS]
    [--checkpoint CHECKPOINT]
    [--gpu GPU]

Arguments:
    -h, --help
        show this help message and exit
    --text_files TEXT_FILES [TEXT_FILES ...]
        The speech transcript files
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
        The speech audio files
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
        The files to save the alignments
    --aligner ALIGNER
        The alignment method to use
    --num_workers NUM_WORKERS
        Number of CPU cores to utilize. Defaults to all cores.
    --checkpoint CHECKPOINT
        The checkpoint to use for neural methods
    --gpu GPU
        The index of the GPU to use for inference. Defaults to CPU.
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

### Evaluate

```
python -m pyfal.evaluate \
    --config <config> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```

Evaluate a model. `<checkpoint>` is the checkpoint file to evaluate and `<gpu>`
is the GPU index.


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
