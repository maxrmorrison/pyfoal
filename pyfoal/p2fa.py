import string
import subprocess
import tempfile
from pathlib import Path

import g2p_en
import pypar
import soundfile

import pyfoal


###############################################################################
# P2FA forced aligner
###############################################################################


class Aligner:
    """P2fa forced aligner"""

    def __init__(self):
        """Aligner constructor"""
        self.hcopy = pyfoal.ASSETS_DIR / 'config'
        self.macros = pyfoal.ASSETS_DIR / 'macros'
        self.model = pyfoal.ASSETS_DIR / 'hmmdefs'
        self.monophones = pyfoal.ASSETS_DIR / 'monophones'

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

            # Remove characters we can't handle
            text = self.lint(text)

            # Run alignment model
            alignment_file = directory / 'alignment.mlf'
            self.viterbi(self.write_words(directory, text),
                         self.write_pronunciation(directory, text),
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
        durations[0] += .0125
        durations = [d * 11000. / 11025. for d in durations]

        # End at audio duration
        durations[-1] = duration - sum(durations[:-1])

        # Update alignment durations
        alignment.update(durations=durations)

        return alignment

    def format(self, directory, audio, script_file):
        """Write HTK arguments and convert data to HTK format"""
        # Save audio to disk
        audiofile = directory / 'sound.wav'
        soundfile.write(audiofile, audio, pyfoal.HTK_SAMPLE_RATE)

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

    def write_pronunciation(self, directory, text):
        """Write the pronunciation dictionary"""
        # Grapheme-to-phoneme conversion
        phonemes = g2p_en.G2p()(text)

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
