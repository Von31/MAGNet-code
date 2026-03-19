#!/bin/bash
# ============================================================
# Download script for  magnet-code model checkpoints, training
# data, and body model files from Google Drive.
#
# Expected directory structure after running:
#   magnet-code/
#   └── data/                   (training/evaluation data)
#
# Usage:  bash scripts/download_dataset.sh
# ============================================================

set -e  # exit on first error

# --- Training / evaluation data ---
echo ">> Downloading data into data/ ..."
gdown --folder https://drive.google.com/drive/folders/1Mu2M6kERuOz0-4oOsUSA4yzUstC0yCKc?usp=drive_link --fuzzy

echo ">> All downloads complete."
