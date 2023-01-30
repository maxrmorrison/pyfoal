# Runs all experiments and evaluations

# Args
# $1 - list of indices of GPUs to use

# Download datasets
python -m pyfoal.data.download

# Setup experiments
python -m pyfoal.data.preprocess --gpu $1
python -m pyfoal.partition

# Train
python -m pyfoal.train --config config/radtts.py --gpus $1

# Evaluate
python -m pyfoal.evaluate --config config/mfa.py
python -m pyfoal.evaluate --config config/p2fa.py
python -m pyfoal.evaluate --config config/radtts.py --checkpoint runs/radtts/00250000.pt
