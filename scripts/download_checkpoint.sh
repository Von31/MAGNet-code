#!/bin/bash
# ============================================================
# Download script for  magnet-code model checkpoints, training
# data, and body model files from Google Drive.
#
# Expected directory structure after running:
#   magnet-code/
#   ├── checkpoints/
#   │   └── magnet_dd100/       (DFOT model checkpoint)
#
# Usage:  bash scripts/download_checkpoint.sh
# ============================================================

set -e  # exit on first error

# --- DFOT model checkpoint (magnet_dd100) ---
echo ">> Downloading DFOT checkpoint into checkpoints/ ..."
gdown --folder https://drive.google.com/drive/folders/1_kQJNJx_GNNbLuVvQbqe42I-fDrbunJZ?usp=drive_link --fuzzy

echo ">> All downloads complete."
