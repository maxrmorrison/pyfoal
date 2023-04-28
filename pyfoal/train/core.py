import contextlib
import functools
import os

import torch

import pyfoal


###############################################################################
# Training interface
###############################################################################


def run(
    datasets,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus=None):
    """Run model training"""
    # Distributed data parallelism
    if gpus and len(gpus) > 1:
        args = (
            datasets,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus)
        torch.multiprocessing.spawn(
            train_ddp,
            args=args,
            nprocs=len(gpus),
            join=True)

    else:

        # Single GPU or CPU training
        train(
            datasets,
            checkpoint_directory,
            output_directory,
            log_directory,
            None if gpus is None else gpus[0])

    # Return path to model checkpoint
    return pyfoal.checkpoint.latest_path(output_directory)


###############################################################################
# Training
###############################################################################


def train(
    datasets,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpu=None):
    """Train a model"""
    # Get DDP rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = None

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(pyfoal.RANDOM_SEED)
    train_loader = pyfoal.data.loader(datasets, 'train', gpu)
    valid_loader = pyfoal.data.loader(datasets, 'valid', gpu)
    test_loader = pyfoal.data.loader(pyfoal.EVALUATION_DATASETS, 'valid', gpu)

    #################
    # Create models #
    #################

    model = pyfoal.Model().to(device)

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = pyfoal.checkpoint.latest_path(checkpoint_directory, '*.pt')

    if path is not None:

        # Load model
        model, optimizer, step = pyfoal.checkpoint.load(path, model, optimizer)

    else:

        # Train from scratch
        step = 0

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank])

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Setup progress bar
    if not rank:
        progress = pyfoal.iterator(
            range(step, pyfoal.STEPS),
            f'Training {pyfoal.CONFIG}',
            initial=step,
            total=pyfoal.STEPS)
    while step < pyfoal.STEPS:

        # Seed sampler
        train_loader.batch_sampler.set_epoch(
            (step + len(train_loader.dataset) - 1) //
            len(train_loader.dataset))

        model.train()
        for batch in train_loader:

            # Unpack batch
            phonemes, audio, priors, mask, phoneme_lengths, frame_lengths, *_ = batch

            with torch.autocast(device.type):

                # Forward pass
                logits = model(
                    phonemes.to(device),
                    audio.to(device),
                    priors.to(device),
                    mask.to(device))

                # Compute loss
                losses = loss(
                    logits,
                    phoneme_lengths.to(device),
                    frame_lengths.to(device))

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                ############
                # Evaluate #
                ############

                if step % pyfoal.LOG_INTERVAL == 0:
                    evaluate_fn = functools.partial(
                        evaluate,
                        log_directory,
                        step,
                        model,
                        gpu)
                    evaluate_fn('train', train_loader)
                    evaluate_fn('valid', valid_loader)
                    evaluate_fn('test', test_loader)

                ###################
                # Save checkpoint #
                ###################

                if step and step % pyfoal.CHECKPOINT_INTERVAL == 0:
                    pyfoal.checkpoint.save(
                        model,
                        optimizer,
                        step,
                        output_directory / f'{step:08d}.pt')

            # Update training step count
            if step >= pyfoal.STEPS:
                break
            step += 1

            # Update progress bar
            if not rank:
                progress.update()

    # Close progress bar
    if not rank:
        progress.close()

    # Save final model
    pyfoal.checkpoint.save(
        model,
        optimizer,
        step,
        output_directory / f'{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, gpu, condition, loader):
    """Perform model evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Setup evaluation metrics
    metrics = pyfoal.evaluate.Metrics()

    # Tensorboard audio and figures
    waveforms, figures = {}, {}

    # Prepare model for inference
    with pyfoal.inference_context(model):

        for i, batch in enumerate(loader):

            # Unpack batch
            (
                phonemes,
                audios,
                priors,
                mask,
                phoneme_lengths,
                frame_lengths,
                stems,
                targets,
                _
            ) = batch

            # Forward pass
            logits = model(
                phonemes.to(device),
                audios.to(device),
                priors.to(device),
                mask.to(device)).detach()

            if condition == 'test':

                # Decode
                alignments = [
                    pyfoal.postprocess(
                        phoneme[:, :phoneme_length],
                        logit[:frame_length, :phoneme_length],
                        audio[:, :pyfoal.convert.frames_to_samples(frame_length)])
                    for phoneme, logit, audio, phoneme_length, frame_length in
                    zip(phonemes, logits, audios, phoneme_lengths, frame_lengths)]

                # Add audio and alignment plot
                if i == 0:
                    iterator = zip(
                        audios[:pyfoal.PLOT_EXAMPLES],
                        logits[:pyfoal.PLOT_EXAMPLES],
                        stems[:pyfoal.PLOT_EXAMPLES],
                        frame_lengths[:pyfoal.PLOT_EXAMPLES],
                        phoneme_lengths[:pyfoal.PLOT_EXAMPLES],
                        alignments[:pyfoal.PLOT_EXAMPLES],
                        targets[:pyfoal.PLOT_EXAMPLES])
                    for (
                        audio,
                        logit,
                        stem,
                        frame_length,
                        phoneme_length,
                        alignment,
                        target
                    ) in iterator:

                        # Add audio
                        samples = pyfoal.convert.frames_to_samples(frame_length)
                        waveforms[f'audio/{stem}'] = audio[:, :samples]

                        # Add alignment figure
                        logit = torch.nn.functional.log_softmax(
                            logit[:frame_length, :phoneme_length],
                            dim=1)
                        logit[logit < -60.] = -60.
                        figures[f'attention/{stem}'] = \
                            pyfoal.plot.logits(logit.cpu())
                        figures[f'alignment/{stem}'] = \
                            pyfoal.plot.alignments(audio, alignment, target)

            else:

                alignments, targets = None, None

            # Update metrics
            metrics.update(
                alignments,
                targets,
                logits,
                phoneme_lengths.to(device),
                frame_lengths.to(device))

            # Stop when we exceed some number of batches
            if i + 1 == pyfoal.LOG_STEPS:
                break

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    pyfoal.write.audio(directory, step, waveforms)
    pyfoal.write.figures(directory, step, figures)
    pyfoal.write.scalars(directory, step, scalars)


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(
    rank,
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus[rank])


@contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)

    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()


###############################################################################
# Utilities
###############################################################################


def loss(logits, phoneme_lengths, frame_lengths):
    # Pad logits
    logits = torch.nn.functional.pad(logits, (1, 0), value=-1)

    total = 0.
    iterator = zip(logits, phoneme_lengths, frame_lengths)
    for logit, phoneme_length, frame_length in iterator:

        # Make ground truth targets
        target_seq = torch.arange(1, phoneme_length + 1).unsqueeze(0)

        # Compute log probabilities
        log_prob = torch.nn.functional.log_softmax(
            logit[:frame_length, :phoneme_length + 1],
            dim=1)

        # Compute CTC loss
        total += torch.nn.functional.ctc_loss(
            log_prob,
            target_seq,
            input_lengths=frame_length[None],
            target_lengths=phoneme_length[None],
            zero_infinity=True)

    # Average
    return total / logits.shape[0]
