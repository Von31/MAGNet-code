#!/bin/bash
# ============================================================
# Download script for  magnet-code model checkpoints, training
# data, and body model files from Google Drive.
#
# Expected directory structure after running:
#   magnet-code/
#   ├── checkpoints/
#   │   └── magnet_dd100/       (DFOT model checkpoint)
#   ├── data/                   (training/evaluation data)
#   └── body_model/             (SMPL-X body model files)
#
# Usage:  bash scripts/download_data_checkpoint.sh
# ============================================================

set -e  # exit on first error

# --- Environment setup ---
eval "$(conda shell.bash hook)"
conda activate mc

# --- 1. DFOT model checkpoint (magnet_dd100) ---
echo ">> [1/3] Downloading DFOT checkpoint into checkpoints/ ..."
gdown --folder https://drive.google.com/drive/folders/1_kQJNJx_GNNbLuVvQbqe42I-fDrbunJZ?usp=drive_link --fuzzy
cd ..

# --- 2. Training / evaluation data ---
echo ">> [2/3] Downloading data into data/ ..."
gdown --folder https://drive.google.com/drive/folders/1Mu2M6kERuOz0-4oOsUSA4yzUstC0yCKc?usp=drive_link --fuzzy

# --- 3. Body model (SMPL-X) ---
echo ">> [3/3] Downloading body model into body_model/ ..."
gdown --folder https://drive.google.com/drive/folders/1SnwhqU96QWkCMtZonfenhpIlusgQYrU_?usp=drive_link --fuzzy

echo ">> All downloads complete."


gdown --folder https://drive.google.com/drive/folders/1_kQJNJx_GNNbLuVvQbqe42I-fDrbunJZ?usp=drive_link --fuzzy