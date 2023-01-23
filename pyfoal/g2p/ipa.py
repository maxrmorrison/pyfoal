import phonemizer


###############################################################################
# International phonetic alphabet (IPA) G2P
###############################################################################


def from_text(text):
    """Convert text to ipa"""
    # Separate words using spaces
    separator = phonemizer.separator.Separator(word=' ')

    # Convert words to IPA phonemes
    return phonemizer.phonemize(
        text,
        language='en-us',
        backend='festival',
        separator=separator,
        strip=True,
        preserve_punctuation=True,
        njobs=4)
