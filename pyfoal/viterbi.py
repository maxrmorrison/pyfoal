import torch

import pyfoal


###############################################################################
# Viterbi decoding
###############################################################################


def decode(phonemes, logits, loudness=None):
    """Get phoneme indices and frame counts from network output"""
    # Normalize
    observation = torch.nn.functional.log_softmax(logits, dim=0)

    # Viterbi decoding is faster on CPU
    observation = observation.cpu()

    # Always start at the first phoneme
    initial = torch.full(
        (observation.shape[1],),
        -float('inf'),
        dtype=observation.dtype)
    initial[0] = 0.

    # Enforce monotonicity
    transition = torch.zeros(
        (observation.shape[1], observation.shape[1]),
        dtype=observation.dtype)
    transition.fill_diagonal_(1.)
    transition[
        torch.arange(len(transition) - 1) + 1,
        torch.arange(len(transition) - 1)] = 1.

    # Allow spaces to optionally be skipped
    if pyfoal.ALLOW_SKIP_SPACE:

        # Find spaces according to phonemes
        space = phonemes[0] == pyfoal.convert.phoneme_to_index('<silent>')

        # Get indices
        spaces = 1 + torch.where(space[1:-1])[0]

        # Uniform probability
        transition[spaces + 1, spaces - 1] = 1.

        # Maybe force skip silence if it's not actually silent
        if not pyfoal.ALLOW_LOUD_SILENCE:
            space[0], space[-1] = False, False
            silence_indices = (
                (loudness.squeeze() > pyfoal.SILENCE_THRESHOLD)[:, None] &
                space[None])
            observation[silence_indices] = -float('inf')

    # Normalize
    transition /= transition.sum(dim=1, keepdim=True)
    transition = torch.log(transition)

    # Viterbi decoding forward pass
    posterior, memory = forward(observation, transition, initial)

    # Enforce alignment between final frame and final phoneme
    posterior[-1] = -float('inf')
    posterior[-1, -1] = 0.

    # Backward pass
    indices = backward(posterior, memory)

    # Count consecutive indices
    return torch.unique_consecutive(indices, return_counts=True)


###############################################################################
# Utilities
###############################################################################


def backward(posterior, memory):
    """Get optimal pass from results of forward pass"""
    # Initialize
    indices = torch.full(
        (posterior.shape[0],),
        torch.argmax(posterior[-1]),
        dtype=torch.int,
        device=posterior.device)

    # Backward
    for t in range(indices.shape[0] - 2, -1, -1):
        indices[t] = memory[t + 1, indices[t + 1]]

    return indices


def forward(observation, transition, initial):
    """Viterbi decoding forward pass"""
    # Initialize
    posterior = torch.zeros_like(observation)
    memory = torch.zeros(
        observation.shape,
        dtype=torch.int,
        device=observation.device)

    # Add prior to first frame
    posterior[0] = observation[0] + initial

    # Forward pass
    for t in range(1, observation.shape[0]):
        step(t, observation, transition, posterior, memory)

    return posterior, memory


def step(index, observation, transition, posterior, memory):
    """One step of the forward pass"""
    probability = posterior[index - 1] + transition

    # Update best so far
    for j in range(observation.shape[1]):
        memory[index, j] = torch.argmax(probability[j])
        posterior[index, j] = \
            observation[index, j] + probability[j, memory[index][j]]
