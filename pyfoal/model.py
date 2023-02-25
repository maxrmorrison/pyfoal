import functools

import torch

import pyfoal


###############################################################################
# Attention-based alignment model
###############################################################################


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Text encoding
        self.key_encoder = TextEncoder()

        # Mel encoding
        self.query_encoder = MelEncoder()

    def forward(self, phonemes, audio, prior=None, mask=None):
        # Compute melspectrogram
        # Input shape: (batch, 1, audio.shape[-1])
        # Output shape: (
        #   batch,
        #   pyfoal.NUM_MELS,
        #   pyfoal.convert.samples_to_frames(audio.shape[-1]))
        mels = pyfoal.data.preprocess.mels.from_audio(audio)

        # Isotropic Gaussian attention
        # Input shape: (
        #   (batch, pyfoal.NUM_MELS, mels.shape[-1]),
        #   (batch, 1, phonemes.shape[-1]))
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
        if prior is not None:
            attention = (
                torch.nn.functional.log_softmax(attention, dim=2) +
                pyfoal.PRIOR_WEIGHT * torch.log(prior + 1e-8))

        # Apply mask
        if mask is not None:
            attention.data.masked_fill_(~mask.to(torch.bool), -float('inf'))

        return attention


###############################################################################
# Utilities
###############################################################################


class MelEncoder(torch.nn.Sequential):

    def __init__(self):
        mels = pyfoal.NUM_MELS
        channels = pyfoal.MEL_ENCODER_WIDTHS
        kernels = pyfoal.MEL_ENCODER_KERNEL_SIZES
        conv_fn = functools.partial(torch.nn.Conv1d, padding='same')
        layers = [
            conv_fn(mels, channels[0], kernel_size=kernels[0]),
            torch.nn.ReLU()]
        prev = channels[0]
        for channel, kernel in zip(channels[1:], kernels[1:-1]):
            layers.extend([
                conv_fn(prev, channel, kernel_size=kernel),
                torch.nn.ReLU()])  
            prev = channel
        layers.append(
            conv_fn(
            channels[-1],
            pyfoal.ATTENTION_WIDTH,
            kernel_size=kernels[-1]))
        super().__init__(*layers)


class TextEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Text embedding
        channels = pyfoal.PHONEME_EMBEDDING_SIZE
        self.embedding = torch.nn.Embedding(
            len(pyfoal.load.phonemes()),
            channels)

        # Text encoding
        conv_fn = functools.partial(torch.nn.Conv1d, padding='same')
        kernel_sizes = pyfoal.TEXT_ENCODER_KERNEL_SIZES
        self.input_stack = torch.nn.Sequential(
            conv_fn(channels, channels, kernel_size=kernel_sizes[0]),
            torch.nn.ReLU(),
            conv_fn(channels, channels, kernel_size=kernel_sizes[1]),
            torch.nn.ReLU(),
            conv_fn(channels, channels, kernel_size=kernel_sizes[2]),
            torch.nn.ReLU())
        self.output_stack = torch.nn.Sequential(
            conv_fn(channels, 2 * channels, kernel_size=kernel_sizes[3]),
            torch.nn.ReLU(),
            conv_fn(2 * channels, pyfoal.ATTENTION_WIDTH, kernel_size=kernel_sizes[4]))

        # Maybe add LSTM
        if pyfoal.LSTM:
            self.lstm = torch.nn.LSTM(
                channels,
                channels // 2,
                batch_first=True,
                bidirectional=True)
    
    def forward(self, phonemes):
        # Encode text
        # Input shape: (batch, 1, phonemes.shape[-1])
        # Output shape: (
        #   batch, pyfoal.PHONEME_EMBEDDING_SIZE, phonemes.shape[-1])
        phonemes = self.embedding(phonemes.squeeze(1)).transpose(1, 2)
        activation = self.input_stack(phonemes)
        if pyfoal.LSTM:
            activation, _ = self.lstm(activation.transpose(1, 2))
            activation = activation.transpose(1, 2)
        return self.output_stack(activation)
