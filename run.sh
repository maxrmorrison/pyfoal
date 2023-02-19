# Runs all experiments and evaluations

# Args
# $1 - index of GPU to use

# Download datasets
python -m pyfoal.data.download --datasets arctic

# Setup experiments
python -m pyfoal.data.preprocess --datasets arctic
python -m pyfoal.partition --datasets arctic

# Train
python -m pyfoal.train --config config/radtts.py --gpus $1

# Evaluate
python -m pyfoal.evaluate --config config/mfa.py
python -m pyfoal.evaluate --config config/p2fa.py
python -m pyfoal.evaluate \
    --config config/radtts.py \
    --checkpoint runs/radtts/00250000.pt \
    --gpu $1
