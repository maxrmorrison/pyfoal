import json
import time

import pyfoal


###############################################################################
# Evaluate
###############################################################################


def datasets(
    datasets=pyfoal.DATASETS,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform evaluation"""
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

    # Start benchmarking
    pyfoal.BENCHMARK = True
    pyfoal.TIMER.reset()
    start_time = time.time()

    # Evaluate each dataset
    samples = 0
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        iterator = pyfoal.iterator(
            pyfoal.data.loader([dataset], 'test'),
            f'Evaluating {pyfoal.CONFIG} on {dataset}')

        # Iterate over test set
        for batch in iterator:

            # Reset file metrics
            file_metrics.reset()

            # Unpack
            _, audio, _, _, _, _, stem, target, text = batch

            # Align
            with pyfoal.time.timer('align'):
                alignment = pyfoal.from_text_and_audio(
                    text[0],
                    audio[0],
                    pyfoal.SAMPLE_RATE,
                    checkpoint=checkpoint,
                    gpu=gpu)

            # Update metrics
            samples += audio.numel()
            args = [alignment], target
            file_metrics.update(*args)
            dataset_metrics.update(*args)
            aggregate_metrics.update(*args)

            # Copy results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
        overall[dataset] = dataset_metrics()
    overall['aggregate'] = aggregate_metrics()

    # Write to json files
    directory = pyfoal.EVAL_DIR / pyfoal.CONFIG
    directory.mkdir(exist_ok=True, parents=True)
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)

    # Turn off benchmarking
    pyfoal.BENCHMARK = False

    # Get benchmarking information
    benchmark = pyfoal.TIMER()
    benchmark['total'] = time.time() - start_time
    seconds = pyfoal.convert.samples_to_seconds(samples)
    benchmark = {
        key: {
            'real-time-factor': value / seconds,
            'samples': samples,
            'samples-per-second': samples / value,
            'seconds': value
        } for key, value in benchmark.items()}

    # Write benchmark to json file
    with open(directory / 'benchmark.json', 'w') as file:
        json.dump(benchmark, file, indent=4)
