import matplotlib.pyplot as plt
import torch

import pyfoal


###############################################################################
# Plots
###############################################################################


def logits(x, alignment=None):
    """Plot the network alignment output"""
    # Maybe highlight selected path
    if alignment is not None:

        # Convert to framewise indices
        hopsize = pyfoal.HOPSIZE / pyfoal.SAMPLE_RATE
        times = hopsize / 2 + torch.arange(len(x)) * hopsize
        phonemes = pyfoal.convert.alignment_to_indices(
            alignment,
            hopsize,
            times=times)
        _, indices = torch.unique_consecutive(
            torch.tensor(phonemes),
            return_inverse=True)
        
        # Highlight
        x[
            torch.arange(len(indices), dtype=torch.long, device=x.device),
            indices
        ] = 10.
    
    # Plot
    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(
        x.T,
        aspect='auto',
        origin='lower',
        interpolation='none')
    plt.colorbar(image, ax=axis)
    plt.xlabel('Frames')
    plt.ylabel('Phonemes')
    plt.tight_layout()

    return figure
