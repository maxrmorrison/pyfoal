# Runs all experiments and evaluations

# Args
# $1 - index of GPU to use

# Download datasets
python -m pyfoal.data.download

# Setup experiments
python -m pyfoal.data.preprocess
python -m pyfoal.partition

# Train (and evaluate)
python -m pyfoal.train --config config/radtts-trf1-mrf9-512-4x512-sil62-interp.py --gpus $1

# Evaluate
python -m pyfoal.evaluate --config config/p2fa.py
python -m pyfoal.evaluate --config config/mfa.py
