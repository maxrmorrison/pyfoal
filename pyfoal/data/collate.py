import torch

import pyfoal


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Batch collation"""
    # Unpack
    phonemes, audios, priors, alignments, text, stems = zip(*batch)

    # Get phoneme lengths
    phoneme_lengths = torch.tensor(
        [phoneme.shape[-1] for phoneme in phonemes],
        dtype=torch.long)
    max_phoneme_length = phoneme_lengths.max().item()

    # Get audio lengths
    audio_lengths = torch.tensor(
        [audio.shape[-1] for audio in audios],
        dtype=torch.long)
    max_audio_length = audio_lengths.max().item()

    # Get frame lengths
    frame_lengths = pyfoal.convert.samples_to_frames(audio_lengths)
    max_frame_length = pyfoal.convert.samples_to_frames(max_audio_length)

    # Get padded tensors for training
    padded_phonemes = torch.zeros(
        (len(phonemes), 1, max_phoneme_length),
        dtype=torch.long)
    padded_audio = torch.zeros((len(audios), 1, max_audio_length))
    padded_priors = torch.zeros((len(priors), max_frame_length, max_phoneme_length))

    # Get sequence mask
    mask = torch.zeros(
        (len(phonemes), max_frame_length, max_phoneme_length),
        dtype=torch.bool)

    # Place batch in padded tensors
    iterator = enumerate(
        zip(
            phonemes,
            audios,
            priors,
            phoneme_lengths,
            audio_lengths,
            frame_lengths))
    for (
        i,
        (phoneme, audio, prior, phoneme_length, audio_length, frame_length)
    ) in iterator:

        # Pad phonemes
        padded_phonemes[i, :, :phoneme_length] = phoneme

        # Pad audio
        padded_audio[i, :, :audio_length] = audio

        # Pad prior
        padded_priors[i, :frame_length, :phoneme_length] = prior

        # Create mask
        mask[i, :frame_length, :phoneme_length] = True

    return (
        padded_phonemes,
        padded_audio,
        padded_priors,
        mask,
        phoneme_lengths,
        frame_lengths,
        stems,
        alignments,
        text)
