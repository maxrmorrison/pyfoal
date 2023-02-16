import contextlib
import functools
import math
import os

import pypar
import torbi
import torch
import torchaudio
import tqdm

import pyfoal


###############################################################################
# Forced alignment
###############################################################################


def from_text_and_audio(
    text,
    audio,
    sample_rate,
    aligner=pyfoal.ALIGNER,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Phoneme-level forced-alignment

    Arguments
        text : string
            The speech transcript
        audio : torch.tensor(shape=(1, samples))
            The speech signal to process
        sample_rate : int
            The audio sampling rate

    Returns
        alignment : pypar.Alignment
            The forced alignment
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        return pyfoal.baselines.mfa.align(text, audio, sample_rate)

    # Penn phonetic forced aligner
    if aligner == 'p2fa':
        return pyfoal.baselines.p2fa.align(text, audio, sample_rate)

    # RAD-TTS neural alignment
    if aligner == 'radtts':

        # Get inference device
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Preprocess
        phonemes, audio = preprocess(text, audio, sample_rate)

        # Infer
        logits = infer(phonemes.to(device), audio.to(device), checkpoint)

        # Postprocess
        return postprocess(audio[0], phonemes[0], logits[0])

    raise ValueError(f'Aligner {aligner} is not defined')


def from_file(
    text_file,
    audio_file,
    aligner=pyfoal.ALIGNER,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Phoneme alignment from audio and text files

    Arguments
        text_file : Path
            The corresponding transcript file
        audio_file : Path
            The audio file to process
        aligner : str
            The alignment method to use
        checkpoint : Path
            The checkpoint to use for neural methods
        gpu : int
            The index of the gpu to perform alignment on for neural methods

    Returns
        alignment : Alignment
            The forced alignment
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        return pyfoal.baselines.mfa.from_file(text_file, audio_file)

    # Penn phonetic forced aligner
    if aligner == 'p2fa':
        return pyfoal.baselines.p2fa.from_file(text_file, audio_file)

    # RAD-TTS neural alignment
    if aligner == 'radtts':

        # Load text
        text = pyfoal.load.text(text_file)

        # Load audio
        audio, sample_rate = pyfoal.load.audio(audio_file)

        # Align
        return from_text_and_audio(
            text,
            audio,
            sample_rate,
            aligner,
            checkpoint,
            gpu)

    raise ValueError(f'Aligner {aligner} is not defined')


def from_file_to_file(
    text_file,
    audio_file,
    output_file,
    aligner=pyfoal.ALIGNER,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform phoneme alignment from files and save to disk

    Arguments
        text_file : Path
            The corresponding transcript file
        audio_file : Path
            The audio file to process
        output_file : Path
            The file to save the alignment
        aligner : str
            The alignment method to use
        checkpoint : Path
            The checkpoint to use for neural methods
        gpu : int
            The index of the gpu to perform alignment on for neural methods
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        pyfoal.baselines.mfa.from_file_to_file(
            text_file,
            audio_file,
            output_file)

    # Penn phonetic forced aligner
    elif aligner == 'p2fa':
        pyfoal.baselines.p2fa.from_file_to_file(
            text_file,
            audio_file,
            output_file)

    # RAD-TTS neural alignment
    elif aligner == 'radtts':

        # Align
        alignment = from_file(text_file, audio_file, aligner, checkpoint, gpu)

        # Save
        alignment.save(output_file)

    else:
        raise ValueError(f'Aligner {aligner} is not defined')


def from_files_to_files(
    text_files,
    audio_files,
    output_files,
    aligner=pyfoal.ALIGNER,
    num_workers=None,
    checkpoint=pyfoal.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform parallel phoneme alignment from many files and save to disk

    Arguments
        text_files : list
            The transcript files
        audio_files : list
            The corresponding speech audio files
        output_files : list
            The files to save the alignments
        aligner : str
            The alignment method to use
        num_workers : int
            Number of CPU cores to utilize. Defaults to all cores.
        checkpoint : Path
            The checkpoint to use for neural methods
        gpu : int
            The index of the gpu to perform alignment on for neural methods
    """
    # Montreal forced aligner
    if aligner == 'mfa':
        pyfoal.baselines.mfa.from_files_to_files(
            text_files,
            audio_files,
            output_files,
            num_workers)

    # Penn phonetic forced aligner
    elif aligner == 'p2fa':
        pyfoal.baselines.p2fa.from_files_to_files(
            text_files,
            audio_files,
            output_files,
            num_workers)

    # RAD-TTS neural alignment
    elif aligner == 'radtts':
        align_fn = functools.partial(
            from_file_to_file,
            checkpoint=checkpoint,
            gpu=gpu)
        for item in zip(text_files, audio_files, output_files):
            align_fn(*item)

    else:
        raise ValueError(f'Aligner {aligner} is not defined')


###############################################################################
# RAD-TTS alignment
###############################################################################


def decode(phonemes, logits):
    """Get phoneme indices and frame counts from network output"""
    # Normalize
    observation = torch.nn.functional.log_softmax(logits, dim=0)

    # Viterbi decoding is faster on CPU
    observation = observation.cpu()

    # Always start at the first phoneme
    initial = torch.full(
        (observation.shape[1],),
        -float('inf'),
        dtype=observation.dtype)
    initial[0] = 0.
    
    # Enforce monotonicity
    transition = torch.zeros(
        (observation.shape[1], observation.shape[1]),
        dtype=observation.dtype)
    transition.fill_diagonal_(1.)
    transition[
        torch.arange(len(transition) - 1) + 1,
        torch.arange(len(transition) - 1)] = 1.
    
    # Allow spaces to optionally be skipped
    spaces = torch.where(
        phonemes[:-2] == pyfoal.convert.phoneme_to_index('<silent>'))
    transition[spaces + 2, spaces] = 1.

    # Normalize
    transition /= transition.sum(dim=1, keepdim=True)
    transition = torch.log(transition)

    # Viterbi decoding forward pass
    posterior, memory = torbi.forward(observation, transition, initial)

    # Enforce alignment between final frame and final phoneme
    posterior[-1] = -float('inf')
    posterior[-1, -1] = 0.

    # Backward pass
    indices = torbi.backward(posterior, memory)

    # Count consecutive indices
    return torch.unique_consecutive(indices, return_counts=True)[1]


def infer(phonemes, audio, checkpoint=pyfoal.DEFAULT_CHECKPOINT):
    """Perform forward pass to retrieve attention alignment"""
    # Maybe cache model
    if (
        not hasattr(infer, 'model') or
        infer.checkpoint != checkpoint or
        infer.device != phonemes.device
    ):
        # Maybe initialize model
        model = pyfoal.Model()

        # Load from disk
        infer.model, *_ = pyfoal.checkpoint.load(checkpoint, model)
        infer.checkpoint = checkpoint
        infer.device = phonemes.device

        # Move model to correct device
        infer.model = infer.model.to(infer.device)

    with inference_context(infer.model):

        # Get prior distribution
        prior = pyfoal.data.preprocess.prior.from_lengths(
            phonemes.shape[-1],
            pyfoal.convert.samples_to_frames(audio.shape[-1])).to(infer.device)

        # Infer
        return infer.model(phonemes, audio, prior)


def postprocess(audio, phonemes, logits):
    """Postprocess logits to produce alignment"""
    # Get per-phoneme frame counts from network output
    counts = decode(logits)

    # Convert phoneme indices to phonemes
    phonemes = pyfoal.convert.indices_to_phonemes(phonemes[0])

    # Get phoneme durations in seconds
    times = torch.cumsum(
        torch.cat(
            (torch.zeros(1, dtype=counts.dtype, device=counts.device),
            counts)),
        dim=0)
    times = pyfoal.convert.frames_to_seconds(times)

    # Match phonemes and start/end times
    alignment = [
        pypar.Word(phoneme, [pypar.Phoneme(phoneme, start, end)])
        for phoneme, start, end in zip(phonemes, times[:-1], times[1:])]

    return pypar.Alignment(alignment)


def preprocess(text, audio, sample_rate):
    """Preprocess text and audio for alignment"""
    # Convert text to IPA
    phonemes = pyfoal.g2p.from_text(text)

    # Resample audio
    audio = resample(audio, sample_rate)

    return phonemes, audio


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def chdir(directory):
    """Context manager for changing the current working directory"""
    previous_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous_directory)


@contextlib.contextmanager
def inference_context(model):
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Automatic mixed precision on GPU
        if device_type == 'cuda':
            with torch.autocast(device_type):
                yield

        else:
            yield

    # Prepare model for training
    model.train()


def iterator(iterable, message, initial=0, total=None):
    """Create a tqdm iterator"""
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=len(iterable) if total is None else total)


def resample(audio, sample_rate, target_rate=pyfoal.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
