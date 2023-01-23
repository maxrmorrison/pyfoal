import phonemizer

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
