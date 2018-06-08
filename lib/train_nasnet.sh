#!/usr/bin/env bash

# Run with:
#       bash scripts/dense_cap_train.sh [dataset] [net] [ckpt_to_init] [data_dir] [step]

set -x
set -e

export PYTHONUNBUFFERED='True'

step=$1
ckpt_path=$2

# For my own experiment usage, just ignore it.
if [ -d '/home/joe' ]; then
    ckpt_path="/home/joe/git/furniture/output/NasNet_big_2/ckpt"
fi

LOG="logs/nasnet_big_${step}_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ ${step} -lt '2' ]
then
time python ./lib/train_net.py \
    --weights ${ckpt_path} \
    --iters 100000 \
    --set EXP_DIR NasNet_focal TRAIN.BATCH_SIZE 5 NET "nasnet" TRAIN.LEARNING_RATE 1e-2 TRAIN.IMG_WIDTH 299 TRAIN.IMG_HEIGHT 299 TRAIN.EXP_DECAY_RATE 0.8 TRAIN.SUMMARY_INTERVAL 6000 TRAIN.EXP_DECAY_STEPS 5000
fi
