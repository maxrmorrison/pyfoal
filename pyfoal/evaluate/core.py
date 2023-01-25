import json

import torch

import pyfoal


###############################################################################
# Evaluate
###############################################################################


def datasets(
    datasets=pyfoal.EVALUATION_DATASETS,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Containers for results
    overall, granular = {}, {}

    # Get metric class
    metric_fn = pyfoal.evaluate.Metrics

    # Per-file metrics
    file_metrics = metric_fn()

    # Per-dataset metrics
    dataset_metrics = metric_fn()

    # Aggregate metrics over all datasets
    aggregate_metrics = metric_fn()

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = pyfoal.iterator(
            pyfoal.data.loader([dataset], 'test'),
            f'Evaluating {pyfoal.CONFIG} pitch quality on {dataset}')

        # Iterate over test set
        for batch in iterator:

            # Reset file metrics
            file_metrics.reset()

            # Unpack
            (
                phonemes,
                audio,
                priors,
                _,
                phoneme_lengths,
                frame_lengths,
                target,
                text,
                stem
            ) = batch

            # Montreal forced aligner
            if pyfoal.METHOD == 'mfa':

                # Align
                alignment = pyfoal.baselines.mfa.from_text_and_audio(
                    text,
                    audio)
                logits = None

            # Penn phonetic forced aligner
            elif pyfoal.METHOD == 'p2fa':

                # Align
                alignment = pyfoal.baselines.p2fa.from_text_and_audio(
                    text,
                    audio)
                logits = None

            # RAD-TTS neural forced aligner
            elif pyfoal.METHOD == 'radtts':

                # Infer
                logits = pyfoal.infer(
                    phonemes.to(device),
                    audio.to(device),
                    checkpoint)

                # Decode
                alignment = pyfoal.postprocess(logits)

            else:

                raise ValueError(f'Method {pyfoal.METHOD} is not defined')

            # Update metrics
            args = logits, phoneme_lengths, frame_lengths, alignment, target
            file_metrics.update(*args)
            dataset_metrics.update(*args)
            aggregate_metrics.update(*args)

            # Copy results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = pyfoal.EVAL_DIR / pyfoal.CONFIG
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)
