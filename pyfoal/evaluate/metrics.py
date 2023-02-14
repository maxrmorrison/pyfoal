import torch

import pyfoal


###############################################################################
# Aggregate metric
###############################################################################


class Metrics:

    def __init__(self):
        self.accuracy = Accuracy()
        self.l1 = L1()
        self.loss = Loss()

    def __call__(self):
        results = {**self.accuracy(), **self.loss()}
        if self.l1.count > 0:
            return {**results, **self.l1()}
        return results

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

        # Update phoneme duration accuracy and error
        if alignments is not None and targets is not None:
            self.accuracy.update(alignments, targets)
            self.l1.update(alignments, targets)

    def reset(self):
        self.l1.reset()
        self.loss.reset()


###############################################################################
# Individual metrics
###############################################################################


class Accuracy:

    def __init__(self, levels=[.01, .005, .0025, .00125]):
        self.levels = levels
        self.reset()
    
    def __call__(self):
        return {
            f'accuracy-{level}': (self.totals[level] / self.count).item()
            for level in self.levels}

    def update(self, alignments, targets):
        for alignment, target in zip(alignments, targets):
            
            # Extract phoneme durations
            predicted_durations = torch.tensor([
                phoneme.duration() for phoneme in alignment.phonemes()])
            target_durations = torch.tensor([
                phoneme.duration() for phoneme in target.phonemes()])
        
            # Update
            for level in self.levels:
                self.total[level] += sum(
                    torch.abs(predicted_durations - target_durations) < level)
                self.count += predicted_durations.numel()

    def reset(self):
        self.count = 0
        self.total = {level: 0. for level in self.levels}


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
            self.total += torch.abs(predicted_durations - target_durations).sum()
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
