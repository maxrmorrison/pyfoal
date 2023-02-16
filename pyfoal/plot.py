import matplotlib.pyplot as plt


###############################################################################
# Plots
###############################################################################


def logits(x):
    """Plot the network alignment output"""
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
