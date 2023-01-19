import torch


###############################################################################
# Attention-based alignment model
###############################################################################


class ConvAttention(torch.nn.Module):

    def __init__(
        self,
        n_mel_channels=80,
        n_text_channels=512):
        super().__init__()
        self.key_encoder = torch.nn.Sequential(
            Masked1dConv(
                n_text_channels,
                2 * n_text_channels,
                kernel_size=3,
                w_init_gain='relu'),
            torch.nn.ReLU(),
            Masked1dConv(n_text_channels*2, n_mel_channels, kernel_size=1))
        self.query_encoder = torch.nn.Sequential(
            Masked1dConv(
                n_mel_channels,
                2 * n_mel_channels,
                kernel_size=3,
                w_init_gain='relu'),
            torch.nn.ReLU(),
            Masked1dConv(n_mel_channels*2, n_mel_channels, kernel_size=1),
            torch.nn.ReLU(),
            Masked1dConv(n_mel_channels, n_mel_channels, kernel_size=1))

    def forward(self, text, mels, mask=None, attention_prior=None):
        # Isotropic Gaussian attention
        # Input shape: (
        #   (batch, mel_channels, frames),
        #   (batch, text_channels, phonemes))
        # Output shape: (batch, mel_channels, frames, phonemes)
        attention = (
            (
                self.query_encoder(mels)[:, :, :, None] -
                self.key_encoder(text)[:, :, None]
            ) ** 2
        )

        # Sum over channels and scale
        # TODO - hparam over temperature
        # Input shape: (batch, mel_channels, frames, phonemes)
        # Output shape: (batch, frames, phonemes)
        attention = -.0005 * attention.sum(1)

        # Maybe add a prior distribution
        # TODO - why log_softmax instead of log?
        if attention_prior is not None:
            attention = (
                torch.nn.functional.log_softmax(attention, dim=2) +
                torch.log(attention_prior + 1e-8))

        # Save copy of logits
        attention_logprob = attention.clone()

        # Apply mask
        if mask is not None:
            attention.data.masked_fill_(~mask, -float('inf'))

        # Softmax along phonemes
        attention = torch.nn.functional.softmax(attention, dim=2)

        # TODO - which to use?
        return attention, attention_logprob


###############################################################################
# Utilities
###############################################################################


class Masked1dConv(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,      # unchanged
        padding=None,  # unchanged
        dilation=1,    # unchanged
        bias=True,     # unchanged
        w_init_gain='linear'):
        super().__init__()
        if padding is None:
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)


# TODO - this is the attention prior distribution used in the above model
def beta_binomial_prior_distribution(
    phoneme_count,
    mel_count,
    scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P - 1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))
