import matplotlib.pyplot as plt
import torch

import pyfoal


###############################################################################
# Plots
###############################################################################


def alignments(audio, alignment, target=None):
    """Plot the word alignment--optionally two"""
    figure, axes = plt.subplots(
        2 + int(target is not None),
        1,
        figsize=(18, 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Plot waveform
    waveform(axes[0], audio)

    # Plot phonemes and dividers of alignment
    phonemes(axes[1], alignment, 2)

    # Maybe plot target alignment dividers in different color
    if target is not None:
        phonemes(axes[2], target, 3, 'red')

    return figure
    

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


###############################################################################
# Utilities
###############################################################################


def phonemes(axis, alignment, rows, color='black'):
    """Plot phoneme alignment on existing axis"""
    # Convert to framewise indices
    hopsize = pyfoal.HOPSIZE / pyfoal.SAMPLE_RATE
    times = (
        hopsize / 2 +
        hopsize * torch.arange(
            pyfoal.convert.seconds_to_frames(alignment.duration())))
    phonemes = pyfoal.convert.alignment_to_indices(
        alignment,
        hopsize,
        times=times)
    phonemes, counts = torch.unique_consecutive(
        torch.tensor(phonemes),
        return_counts=True)
    phonemes = pyfoal.convert.indices_to_phonemes(phonemes)
    indices = torch.cat(
        (torch.zeros(1, dtype=torch.int), torch.cumsum(counts, 0)))
    starts, ends = indices[:-1], indices[1:]

    # Get silent/not-silent classification
    times, values = [], []
    for start, end, phoneme in zip(starts, ends, phonemes):
        times.append(torch.linspace(start, end, (end - start).item(), dtype=torch.int))
        if phoneme == '<silent>':
            values.append(torch.zeros(end - start))
        else:
            values.append(torch.ones(end - start))

    # Plot silent/not-silent
    axis.plot(torch.cat(times), torch.cat(values), color='white', linewidth=.5)
    axis.set_axis_off()

    # Plot phoneme text and dividers
    count = 0
    for start, end, phoneme in zip(starts, ends, phonemes):
        if count > 0 or phoneme != '':
            axis.axvline(
                count,
                color=color,
                linewidth=.5,
                ymin=0,
                ymax=rows,
                clip_on=False,
                linestyle='--')
            axis.text(
                count + .5 * (end - start) - .4 * len(phoneme),
                .5,
                phoneme,
                fontsize=7,
                color=color)
        count += end - start


def waveform(axis, x):
    """Plot waveform on existing axis"""
    time = (
        1. / (2 * pyfoal.SAMPLE_RATE) +
        torch.arange(x.shape[-1]) / pyfoal.SAMPLE_RATE)
    axis.plot(time, x.squeeze(), color='black', linewidth=.5)
    axis.set_axis_off()
    axis.set_ylim([-1., 1.])
