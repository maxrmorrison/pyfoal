import phonemizer
import torch
import tqdm

import pyfoal


###############################################################################
# International phonetic alphabet (IPA) G2P
###############################################################################


def from_text(text):
    """Convert text to ipa"""
    # Maybe make into a list
    if isinstance(text, str):
        text = [text]
        unpack = True
    else:
        unpack = False
        
    # Convert to phonemes
    phonemes = phonemizer.phonemize(
        text,
        language='en-us',
        backend='espeak',
        separator=phonemizer.separator.Separator(word=' '),
        strip=True,
        preserve_punctuation=True)
    
    # Convert to indices
    indices = [
        pyfoal.convert.phonemes_to_indices(phoneme) for phoneme in phonemes]

    # Convert to torch
    indices = [torch.tensor(index) for index in indices]

    return indices[0] if unpack else indices


def from_files_to_files(text_files, output_files):
    """Convert text on disk to phonemes and save"""
    batch_size = pyfoal.ESPEAK_BATCH_SIZE
    for i in tqdm.tqdm(range(0, len(text_files), batch_size), desc='Preprocessing', dynamic_ncols=True):

        # Get text
        text = []
        for file in text_files[i:i + batch_size]:
            text.append(pyfoal.load.text(file))
        
        # Convert to phoneme indices
        indices = from_text(text)

        # Save
        for index, file in zip(indices, output_files[i:i + batch_size]):
            torch.save(index, file)
