import functools
import multiprocessing as mp
import os
import string
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pypar
import soundfile
import torchaudio

import pyfoal


###############################################################################
# Constants
###############################################################################


# The sampling rate that P2FA uses
SAMPLE_RATE = 11025


###############################################################################
# Penn phonetic forced aligner
###############################################################################


def from_text_and_audio(text, audio, sample_rate=pyfoal.SAMPLE_RATE):
    """Align text and audio using P2FA"""
    # Get duration in seconds at original sample rate
    duration = audio.shape[-1] / sample_rate

    # Maybe resample
    audio = pyfoal.resample(audio, sample_rate, SAMPLE_RATE)

    # Cache aligner
    if not hasattr(from_text_and_audio, 'aligner'):
        from_text_and_audio.aligner = Aligner()

    # Align
    return from_text_and_audio.aligner(text, audio, duration)


def from_file(text_file, audio_file):
    """Align text and audio on disk using P2FA"""
    # Load text
    text = pyfoal.load.text(text_file)

    # Load audio
    audio, sample_rate = torchaudio.load(audio_file)

    # Align
    return from_text_and_audio(text, audio, sample_rate)


def from_file_to_file(text_file, audio_file, output_file):
    """Align text and audio on disk using P2FA and save"""
    from_file(text_file, audio_file).save(output_file)


def from_files_to_files(
    text_files,
    audio_files,
    output_files,
    num_workers=None):
    """Align many text and audio files on disk using P2FA and save"""
    # Default to using all cpus
    if num_workers is None:
        num_workers = max(min(len(text_files) // 2, os.cpu_count() // 2), 1)

    # Launch multiprocessed P2FA alignment
    align_fn = functools.partial(from_file_to_file)
    iterator = zip(text_files, audio_files, output_files)
    with mp.get_context('spawn').Pool(num_workers) as pool:
        pool.starmap(align_fn, iterator)


###############################################################################
# P2FA forced aligner
###############################################################################


class Aligner:
    """P2fa forced aligner"""

    def __init__(self):
        """Aligner constructor"""
        self.hcopy = pyfoal.ASSETS_DIR / 'p2fa' / 'config'
        self.macros = pyfoal.ASSETS_DIR / 'p2fa' / 'macros'
        self.model = pyfoal.ASSETS_DIR / 'p2fa' / 'hmmdefs'
        self.monophones = pyfoal.ASSETS_DIR / 'p2fa' / 'monophones'

        punctuation = [s for s in string.punctuation + '”“—' if s != '-']
        self.punctuation_table = str.maketrans('-', ' ', ''.join(punctuation))

    def __call__(self, text, audio, duration):
        """Retrieve the forced alignment"""
        # Alignment artifacts are placed in temporary storage and cleaned-up
        # after alignment is complete
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            # HTK script path
            script_file = directory / 'test.scp'

            # Preprocess
            self.format(directory, audio, script_file)

            # Grapheme-to-phoneme
            text, phonemes = pyfoal.g2p.from_text(
                text,
                to_indices=False,
                remove_prominence=False)

            # Use P2FA silence token
            for i in range(len(phonemes)):
                if phonemes[i] == '<silent>':
                    phonemes[i] = ' '

            # Run alignment model
            alignment_file = directory / 'alignment.mlf'
            self.viterbi(self.write_words(directory, text),
                         self.write_pronunciation(directory, text, phonemes),
                         directory,
                         script_file,
                         alignment_file)

            # Alignment rate and offset correction
            alignment = self.correct_alignment(alignment_file, duration)

        return alignment

    ###########################################################################
    # Utilities
    ###########################################################################

    def correct_alignment(self, alignment_file, duration):
        """Correct alignment rate and offset"""
        # Load alignment
        alignment = pypar.Alignment(alignment_file)

        # Retrieve phoneme durations
        durations = [p.duration() for p in alignment.phonemes()]

        # Constant offset and rate correction
        # TODO - verify
        durations[0] += .0125
        durations = [d * 11000. / 11025. for d in durations]

        # End at audio duration
        durations[-1] = duration - sum(durations[:-1])

        # Update alignment durations
        alignment.update(durations=durations)

        # Change silence token
        for i in range(len(alignment)):
            if str(alignment[i]) == 'sp':
                alignment[i].word = pypar.SILENCE
        for i in range(len(alignment.phonemes())):
            if str(alignment.phonemes()[i]) == 'sp':
                alignment.phonemes()[i].phoneme = pypar.SILENCE

            # Remove prominence markings
            alignment.phonemes()[i].phoneme = ''.join(
                c for c in str(alignment.phonemes()[i]) if not c.isdigit())

        return alignment

    def format(self, directory, audio, script_file):
        """Write HTK arguments and convert data to HTK format"""
        # Save audio to disk
        audiofile = directory / 'sound.wav'
        soundfile.write(
            str(audiofile),
            audio.cpu().squeeze().numpy().astype(float),
            SAMPLE_RATE)

        # Save HTK process metadata
        code_file = directory / 'codetr.scp'
        plp_file = directory / 'tmp.plp'
        with open(code_file, 'w') as file:
            file.write(f'{audiofile} {plp_file}\n')
        with open(script_file, 'w') as file:
            file.write(f'{plp_file}\n')

        # HTK preprocessing call
        subprocess.Popen(
            ['HCopy', '-T', '1', '-C', self.hcopy, '-S', code_file],
            stdout=subprocess.DEVNULL)

    def split_phonemes(self, phonemes):
        """Split phoneme list into words"""
        word, words = [], []
        for phoneme in phonemes:

            # Add next phoneme
            if phoneme == ' ' and word:
                words.append(word)
                word = []
            else:
                word.append(phoneme)

        # Handle final word
        if word:
            words.append(word)

        return words

    def viterbi(self, words, dictionary, directory, script_file, output):
        """Run viterbi decoding to align"""
        args = ['HVite', '-T', '1', '-a', '-m',
                '-I', words,
                '-H', self.macros,
                '-H', self.model,
                '-S', script_file,
                '-i', output,
                '-p', '0.',
                '-s', '5.',
                dictionary,
                self.monophones]
        with open(directory / 'aligned.results', 'w') as file:
            subprocess.Popen(args, stdout=file).wait()

    def write_pronunciation(self, directory, text, phonemes):
        """Write the pronunciation dictionary"""
        # Convert to HTK dictionary format
        iterator = zip(text.upper().split(), self.split_phonemes(phonemes))
        lines = ['{}  {}\n'.format(w, ' '.join(p)) for w, p in iterator]
        lines.append('sp  sp\n')

        # Write HTK dictionary
        filename = directory / 'dictionary'
        with open(filename, 'w') as file:
            for line in sorted(lines):
                file.write(line)

        return filename

    def write_words(self, directory, text):
        """Write the mlf file containing the words to align"""
        filename = directory / 'tmp.mlf'
        with open(filename, 'w') as file:

            # File header
            file.write('#!MLF!#\n')
            file.write('"*/tmp.lab"\n')

            # Write words with spaces in between
            for word in text.upper().split():
                file.write('sp\n')
                file.write(f'{word}\n')

            file.write('sp\n')

        return filename
