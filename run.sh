#!/bin/bash

set -euo pipefail

# Main Norman GraCE-VAE experiment

DEVICE="cuda:0"
MODEL="cmvaegnn"
MODE=15
MX_ALPHA=8
MX_BETA=2
MX_TEMP=4
LMBDA=1e-4
EPOCH=100
GRADCLIP=True

for seed in 1 2 3 4 5 6 7 8 9 10
do
  echo "Running ${MODEL} mode ${MODE} for seed ${seed}"

  python -W ignore src/run.py \
    --device "${DEVICE}" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --randomseed "${seed}" \
    --mxAlpha "${MX_ALPHA}" \
    --mxBeta "${MX_BETA}" \
    --mxTemp "${MX_TEMP}" \
    --lmbda "${LMBDA}" \
    --epoch "${EPOCH}" \
    --gradclip "${GRADCLIP}" 
done
