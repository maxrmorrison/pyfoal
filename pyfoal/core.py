import os
import shutil
import string
import subprocess
import tempfile
import uuid

import g2p_en

import spa


###############################################################################
# Constants
###############################################################################


SAMPLE_RATE = 16000


###############################################################################
# Forced alignment
###############################################################################


def align(audio, text, htk_directory, tmpdir=None):
    """Phoneme-level forced-alignment with HTK

    Arguments
        audio : torch.tensor(shape=(1, time))
            The speech signal to process
        text : string
            The corresponding transcript
        htk_directory : string
            Path to htk binaries
        tmpdir : string or None
            Directory to save temporary values. If None, uses system default.

    Returns
        alignment : Alignment
            The forced alignment
    """
    # Cache aligner
    if not hasattr(align, 'aligner'):
        align.aligner = Aligner()

    # Perform forced alignment
    return align.aligner(audio, text, tmpdir)


def from_file(audio_file, text_file, htk_directory, tmpdir=None):
    """Phoneme alignment from audio and text files

    Arguments
        audio_file : string
            The audio file to process
        text_file : string
            The corresponding transcript file
        htk_directory : string
            Path to htk binaries
        tmpdir : string or None
            Directory to save temporary values. If None, uses system default.

    Returns
        alignment : Alignment
            The forced alignment
    """
    # Load audio
    audio = pyfoal.load.audio(audio_file)

    # Load text
    text = pyfoal.load.text(text_file)

    # Align
    return align(audio, text, tmpdir)


def from_file_to_file(audio_file,
                      text_file,
                      output_file,
                      htk_directory,
                      tmpdir=None):
    """Perform phoneme alignment from files and save to disk

    Arguments
        audio_file : string
            The audio file to process
        text_file : string
            The corresponding transcript file
        output_file : string
            The file to save the alignment
        htk_directory : string
            Path to htk binaries
        tmpdir : string or None
            Directory to save temporary values. If None, uses system default.
    """
    # Align and save
    from_file(audio_file, text_file, tmpdir).save(output_file)


###############################################################################
# Forced aligner object
###############################################################################


class Aligner:
    """P2fa forced aligner"""

    def __init__(self, htk_directory):
        """Aligner constructor

        Arguments
            htk_directory : string
                Path to htk binaries
        """
        self.hcopy = os.path.join(htk_directory, 'config')
        self.macros = os.path.join(htk_directory, 'macros')
        self.model = os.path.join(htk_directory, 'hmmdefs')
        self.monophones = os.path.join(htk_directory, 'monophones')

        punctuation = [s for s in string.punctuation + '”“—' if s != '-']
        self.punctuation_table = str.maketrans('-', ' ', ''.join(punctuation))

    def __call__(self, audio, text, tmpdir=None):
        """Retrieve the forced alignment"""
        if tmpdir is None:
            tmpdir = os.path.join(tempfile.gettempdir(), 'pyfoal')

        # Use a unique directory on each call to allow multiprocessing
        tmpdir = os.path.join(tmpdir, str(uuid.uuid4()))

        # Make sure directory exists
        os.makedirs(tmpdir, exist_ok=True)

        try:

            # HTK script path
            script_file = os.path.join(tmpdir, 'test.scp')

            # Preprocess
            self.format(tmpdir, audio, script_file)

            # Remove characters we can't handle
            text = self.lint(text)

            output_file = os.path.join(tmpdir, 'alignment.mlf')

            # Run alignment model
            self.viterbi(self.write_words(tmpdir, text),
                         self.write_pronunciation(tmpdir, text),
                         tmpdir,
                         script_file,
                         output_file)

            # Retrieve alignment from file
            return spa.Alignment(output_file)

        finally:

            # Remove intermediate features
            shutil.rmtree(tmpdir)

    ###########################################################################
    # Utilities
    ###########################################################################

    def format(self, tmpdir, audio, script_file):
        """Write HTK arguments and convert data to HTK format"""
        # Save audio to disk
        audiofile = os.path.join(tmpdir, 'sound.wav')
        pyfoal.save.audio(audiofile, audio)

        # Save HTK process metadata
        code_file = os.path.join(tmpdir, 'codetr.scp')
        plp_file = os.path.join(tmpdir, 'tmp.plp')
        with open(code_file, 'w') as file:
            file.write(audiofile + ' ' + plp_file + '\n')
        with open(script_file, 'w') as file:
            file.write(plp_file + '\n')

        # HTK preprocessing call
        subprocess.call(
            ['HCopy', '-T', '1', '-C', self.hcopy, '-S', code_file],
            stdout=subprocess.DEVNULL)

    def lint(self, text):
        """Preprocess text for aligner"""
        # Remove newlines, tabs, and extra whitespace
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')

        # Convert numbers to text
        text = g2p_en.expand.normalize_numbers(text)

        # Remove punctuation
        return text.translate(self.punctuation_table)

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

    def viterbi(self, words, dictionary, tmpdir, script_file, output):
        """Run viterbi decoding to align"""
        output_file = os.path.join(tmpdir, 'aligned.results')
        os.system(
            'HVite -T 1 -a -m -I ' + words + ' -H ' + self.macros + ' -H ' + \
            self.model + ' -S ' + script_file + ' -i ' + output + \
            ' -p 0.0 -s 5.0 ' + dictionary + ' ' + self.monophones + ' > ' + \
            output_file + ' 2>&1')

    def write_pronunciation(self, tmpdir, text):
        """Write the pronunciation dictionary"""
        # Grapheme-to-phoneme conversion
        phonemes = g2p_en.G2p()(text)

        # Convert to HTK dictionary format
        iterator = zip(text.upper().split(), self.split_phonemes(phonemes))
        lines = ['{}  {}\n'.format(w, ' '.join(p)) for w, p in iterator]
        lines.append('sp  sp\n')

        # Write HTK dictionary
        filename = os.path.join(tmpdir, 'dictionary')
        with open(filename, 'w') as file:
            for line in sorted(lines):
                file.write(line)

        return filename

    def write_words(self, tmpdir, text):
        """Write the mlf file containing the words to align"""
        filename = os.path.join(tmpdir, 'tmp.mlf')
        with open(filename, 'w') as file:

            # File header
            file.write('#!MLF!#\n')
            file.write('"*/tmp.lab"\n')

            # Write words with spaces in between
            for word in text.upper().split():
                file.write('sp\n')
                file.write(word + '\n')

            file.write('sp\n')

        return filename
