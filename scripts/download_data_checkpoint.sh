#!/bin/bash
# ============================================================
# Download script for  magnet-code model checkpoints, training
# data, and body model files from Google Drive.
#
# Expected directory structure after running:
#   magnet-code/
#   ├── dfot/
#   │   └── magnet_dd100/       (DFOT model checkpoint)
#   ├── vqvae/
#   │   └── magnet_dd100/
#   └── data/             (training/evaluation data)
#
# Usage:  bash scripts/download_data_checkpoint.sh
# ============================================================

set -e  # exit on first error

# --- Environment setup ---
eval "$(conda shell.bash hook)"
conda activate mc

# --- 1. Pose Vqvae and DFOT model checkpoints ---
echo ">> [1/2] Downloading Pose Vqvae and DFOT checkpoint into checkpoints/ ..."
gdown --folder https://drive.google.com/drive/folders/1_kQJNJx_GNNbLuVvQbqe42I-fDrbunJZ?usp=drive_link --fuzzy
cd ..

# --- 2. Preprocessed data ---
echo ">> [2/2] Downloading data into data/ ..."
gdown --folder https://drive.google.com/drive/folders/1Mu2M6kERuOz0-4oOsUSA4yzUstC0yCKc?usp=drive_link --fuzzy


echo ">> All downloads complete."


