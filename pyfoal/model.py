import functools

import numpy as np
import torch

import pyfoal


###############################################################################
# Attention-based alignment model
###############################################################################


class Model(torch.nn.Module):

    def __init__(
        self,
        n_mel_channels=80,
        n_text_channels=512):
        super().__init__()
        # Text embedding
        text_channels = pyfoal.PHONEME_EMBEDDING_SIZE
        self.embedding = torch.nn.Embedding(
            len(pyfoal.PHONEMES),
            text_channels)

        # Text encoding
        conv_fn = functools.partial(torch.nn.Conv1d, padding='same')
        self.key_encoder = torch.nn.Sequential(
            conv_fn(text_channels, text_channels, kernel_size=5),
            torch.nn.ReLU(),
            conv_fn(text_channels, text_channels, kernel_size=5),
            torch.nn.ReLU(),
            conv_fn(text_channels, text_channels, kernel_size=5),
            torch.nn.ReLU(),
            conv_fn(n_text_channels, 2 * n_text_channels, kernel_size=3),
            torch.nn.ReLU(),
            conv_fn(2 * n_text_channels, pyfoal.NUM_MELS, kernel_size=1))

        # Mel encoding
        self.query_encoder = torch.nn.Sequential(
            conv_fn(pyfoal.NUM_MELS, 2 * pyfoal.NUM_MELS, kernel_size=3),
            torch.nn.ReLU(),
            conv_fn(2 * pyfoal.NUM_MELS, pyfoal.NUM_MELS, kernel_size=1),
            torch.nn.ReLU(),
            conv_fn(pyfoal.NUM_MELS, pyfoal.NUM_MELS, kernel_size=1))

    def forward(self, phonemes, audio, mask=None, attention_prior=None):
        # Compute melspectrogram
        # Input shape: (batch, 1, audio.shape[-1])
        # Output shape: (
        #   batch,
        #   pyfoal.NUM_MELS,
        #   pyfoal.convert.samples_to_frames(audio.shape[-1]))
        mels = pyfoal.data.preprocess.mels.from_audio(audio)

        # Encode text
        # Input shape: (batch, 1, phonemes.shape[-1])
        # Output shape: (
        #   batch, pyfoal.PHONEME_EMBEDDING_SIZE, phonemes.shape[-1])
        phonemes = self.embedding(phonemes.squeeze(1)).transpose(1, 2)

        # Isotropic Gaussian attention
        # Input shape: (
        #   (batch, pyfoal.NUM_MELS, mels.shape[-1]),
        #   (batch, pyfoal.PHONEME_EMBEDDING_SIZE, phonemes.shape[-1]))
        # Output shape: (
        #   batch, pyfoal.NUM_MELS, mels.shape[-1], phonemes.shape[-1])
        attention = (
            (
                self.query_encoder(mels)[:, :, :, None] -
                self.key_encoder(phonemes)[:, :, None]
            ) ** 2
        )

        # Sum over channels and scale
        # Input shape: (
        #   batch, pyfoal.NUM_MELS, mels.shape[-1], phonemes.shape[-1])
        # Output shape: (batch, mels.shape[-1], phonemes.shape[-1])
        attention = -pyfoal.TEMPERATURE * attention.sum(1)

        # Maybe add a prior distribution
        if attention_prior is not None:
            attention = (
                torch.nn.functional.log_softmax(attention, dim=2) +
                torch.log(attention_prior + 1e-8))

        # Apply mask
        if mask is not None:
            attention.data.masked_fill_(~mask.to(torch.bool), -float('inf'))

        return attention
