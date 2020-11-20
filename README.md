# Python forced alignment

This is a modified implementation of the Penn Phonetic Forced Aligner (P2FA).
Relative to the original implementation, this repo provides the following.
 - Support for Python 3
 - Support for performing forced alignment both in Python and on the
   command-line
 - Support for multiprocessing. Previous implementations attempt to write
   to the same temporary files when multiprocessing.
 - Fewer alignment failures due to, e.g., out-of-vocabulary (OOV) words or
   punctuation
 - Direct integration with [`pypar`](https://github.com/maxrmorrison/pypar),
   a feature-rich phoneme alignment representation.
 - Clean and documented code


### Installation

##### Hidden Markov Model Toolkit (HTK)
`pyfoal` depends on the [HTK](http://htk.eng.cam.ac.uk/) and has been
tested on Mac OS and Linux using HTK version 3.4.0. There are known issues in
using version 3.4.1 on Linux. HTK is released under a license that prohibits
redistribution, so you must install HTK yourself and verify that the commands
`HCopy` and `HVite` are available as system-wide binaries. After downloading
HTK, I use the following for installation on Linux.

```
./configure --disable-hslab
make all
make install
```

For more help with HTK installation, see notes by
[Jaekoo Kang](https://github.com/jaekookang/p2fa_py3#install-htk) and
[Steve Rubin](https://github.com/ucbvislab/p2fa-vislab#install-htk-34-note-341-will-not-work-get-htk-here).


##### Python dependencies

Clone this repo and run `pip install -e pyfoal/`.


### Usage


##### Force-align audio and text from loaded audio and text

```
alignment = pyfoal.align(audio, sample_rate, text)
```

`audio` must be a `torch.tensor` with shape `(1, samples)`.


##### Force-align audio and text from files

```
alignment = pyfoal.from_file(audio_file, text_file)
```


##### Specifying where to store temporary files

By default, `pyfoal` will write temporary data to a system default location
determined by the built-in `tempfile` module. You can override this location as
follows.

```
pyfoal.align(..., tmpdir='/path/to/store/tmp/files/')
pyfoal.from_file(..., tmpdir='/path/to/store/tmp/files/')
```


##### Command-line interface
TODO - document CLI
