import string

import g2p_en


###############################################################################
# Carnegie Mellon (CMU) G2P
###############################################################################


def from_text(text):
    """Convert text to cmu"""
    # Remove newlines, tabs, and extra whitespace
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')

    # Convert numbers to text
    text = g2p_en.expand.normalize_numbers(text)

    # Remove punctuation
    punctuation = [s for s in string.punctuation + '”“—' if s != '-']
    text = text.translate(str.maketrans('-', ' ', ''.join(punctuation)))

    # Grapheme-to-phoneme conversion
    return g2p_en.G2p()(text)
