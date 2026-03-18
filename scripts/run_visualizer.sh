#!/bin/bash

DATA_DIR=${1:-outputs/dfot/dd100}

CUDA_VISIBLE_DEVICES=0 python libs/viz/visualizer.py --data_dir "$DATA_DIR"
