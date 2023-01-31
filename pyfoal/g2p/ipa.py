import phonemizer
import torch

import pyfoal


###############################################################################
# International phonetic alphabet (IPA) G2P
###############################################################################


def from_text(text):
    """Convert text to ipa"""
    return phonemizer.phonemize(
        text,
        language='en-us',
        backend='espeak',
        separator=phonemizer.separator.Separator(word=' '),
        strip=True,
        preserve_punctuation=True,
        njobs=pyfoal.NUM_WORKERS)


def from_files_to_files(text_files, output_files):
    """Convert text on disk to phonemes and save"""
    # Batch
    batch_size = pyfoal.ESPEAK_BATCH_SIZE
    for i in range(0, len(text_files), batch_size):
            
        # Get text
        text = []
        for text_file in text_files[i:i + batch_size]:
            text.append(pyfoal.load.text(text_file))
    
        # Convert to phonemes
        phonemes = from_text(text)

        # Convert to indices
        indices = [
            pyfoal.convert.phonemes_to_indices(phoneme)
            for phoneme in phonemes]

        # Save
        for index, output_file in zip(indices, output_files[i:i + batch_size]):
            torch.save(torch.tensor(index, dtype=torch.long), output_file)
