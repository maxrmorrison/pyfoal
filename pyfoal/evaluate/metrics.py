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
        results = {}
        if self.loss.count > 0:
            results = self.loss()
        if self.accuracy.count > 0:
            results = {**results, **self.accuracy()}
        if self.l1.count > 0:
            results = {**results, **self.l1()}
        return results

    def update(
        self,
        alignments,
        targets,
        logits=None,
        phoneme_lengths=None,
        frame_lengths=None):
        # Update phoneme duration accuracy and error
        self.accuracy.update(alignments, targets)
        self.l1.update(alignments, targets)

        # Update loss
        if logits is not None:
            self.loss.update(logits, phoneme_lengths, frame_lengths)

    def reset(self):
        self.accuracy.reset()
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

            if alignment is None or target is None:
                continue

            # Extract phoneme durations
            predicted_durations = torch.tensor([
                phoneme.duration() for phoneme in alignment.phonemes()
                if str(phoneme) != '<silent>'])
            target_durations = torch.tensor([
                phoneme.duration() for phoneme in target.phonemes()
                if str(phoneme) != '<silent>'])

            # Maybe update
            if len(predicted_durations) == len(target_durations):
                for level in self.levels:
                    self.totals[level] += sum(
                        torch.abs(predicted_durations - target_durations) < level)
                    self.count += predicted_durations.numel()

    def reset(self):
        self.count = 0
        self.totals = {level: 0. for level in self.levels}


class L1:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'l1': (self.total / self.count).item()}

    def update(self, alignments, targets):
        for alignment, target in zip(alignments, targets):

            if alignment is None or target is None:
                continue

            # Extract phoneme durations
            predicted_durations = torch.tensor([
                phoneme.duration() for phoneme in alignment.phonemes()
                if str(phoneme) != '<silent>'])
            target_durations = torch.tensor([
                phoneme.duration() for phoneme in target.phonemes()
                if str(phoneme) != '<silent>'])

            # Maybe update
            if len(predicted_durations) == len(target_durations):
                self.total += torch.abs(
                    predicted_durations - target_durations).sum()
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
