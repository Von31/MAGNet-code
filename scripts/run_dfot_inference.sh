#!/bin/bash

CONFIG=${1:-configs/inference/dfot/dd100.yaml}

CUDA_VISIBLE_DEVICES=0 python -m libs.inference.dfot_inference --config "$CONFIG"
