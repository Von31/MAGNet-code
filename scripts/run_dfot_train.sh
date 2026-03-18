#!/bin/bash

GPU=${1:-0}
CONFIG=${2:-configs/train/dfot/dd100.yaml}

CUDA_VISIBLE_DEVICES=$GPU python -m libs.train.dfot_train --config "$CONFIG"
