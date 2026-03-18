#!/bin/bash

GPU=${1:-0}
CONFIG=${2:-configs/train/vqvae/dd100.yaml}

CUDA_VISIBLE_DEVICES=$GPU python -m libs.train.vqvae_train --config "$CONFIG"
