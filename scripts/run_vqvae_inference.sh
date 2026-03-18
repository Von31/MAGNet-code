#!/bin/bash

CONFIG=${1:-configs/inference/vqvae/dd100.yaml}

CUDA_VISIBLE_DEVICES=0 python -m libs.inference.vqvae_inference --config "$CONFIG"
