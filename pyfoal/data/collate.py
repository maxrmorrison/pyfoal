import torch

import pyfoal


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Batch collation"""
    # Unpack
    texts, audios, bounds, stems = zip(*batch)

    # Get text lengths
    text_lengths = torch.tensor(
        [text.shape[-1] for text in texts],
        dtype=torch.long)

    # Get audio lengths
    audio_lengths = torch.tensor(
        [audio.shape[-1] for audio in audios],
        dtype=torch.long)

    # Maybe get true word bounds in frames
    padded_text = torch.zeros((len(texts), 1, max_frame_length))
    padded_scores = torch.zeros((len(scores), 1, max_output_length))
    padded_bounds = torch.zeros(
        (len(word_bounds), 2, max_word_length),
        dtype=torch.long)
