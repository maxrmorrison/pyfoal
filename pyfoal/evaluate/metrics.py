import torch

import pyfoal


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.l1 = L1()
        self.loss = Loss()

    def __call__(self):
        return self.l1() | self.loss()

    def update(
        self,
        logits,
        phoneme_lengths,
        frame_lengths,
        alignments=None,
        targets=None):
        # Detach from graph
        logits = logits

        # Update loss
        self.loss.update(logits.detach(), phoneme_lengths, frame_lengths)

        # Update phoneme duration rmse
        if alignments is not None and targets is not None:
            self.l1.update(alignments, targets)

    def reset(self):
        self.l1.reset()
        self.loss.reset()


###############################################################################
# Individual metrics
###############################################################################


class L1:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'l1': (self.total / self.count).item()}

    def update(self, alignments, targets):
        for alignment, target in zip(alignments, targets):

            # Extract phoneme durations
            predicted_durations = torch.tensor([
                phoneme.duration() for phoneme in alignment.phonemes()])
            target_durations = torch.tensor([
                phoneme.duration() for phoneme in target.phonemes()])

            # Update
            self.total += torch.abs(predicted_durations - target_durations)
            self.count += predicted_durations.numel()

    def reset(self):
        self.count = 0
        self.total = 0.


class Loss:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'loss': (self.total / self.count).item()}

    def update(self, logits, phoneme_lengths, frame_lengths):
        self.total += pyfoal.train.loss(logits, phoneme_lengths, frame_lengths)
        self.count += logits.shape[0]

    def reset(self):
        self.count = 0
        self.total = 0.

