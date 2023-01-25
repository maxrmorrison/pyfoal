import matplotlib.pyplot as plt


def logits(x):
    """Plot the network alignment output"""
    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(x, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(image, ax=axis)
    plt.xlabel('Phonemes')
    plt.ylabel('Frames')
    plt.tight_layout()
    return figure
