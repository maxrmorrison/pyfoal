
# Runs training and evaluation of models

# Args
# $1 - list of indices of GPUs to use

# Download datasets
python -m pyfoal.data.download

# Setup experiments
python -m pyfoal.data.preprocess
python -m pyfoal.data.partition

# Train and evaluate experiments
python -m pyfoal.train --config config/radtts.py --gpus $1

# Evaluate baselines
python -m pyfoal.evaluate --config config/mfa.py
python -m pyfoal.evaluate --config config/p2fa.py
