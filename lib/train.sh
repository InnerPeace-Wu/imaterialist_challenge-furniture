#!/usr/bin/env bash

# Run with:
#       bash scripts/dense_cap_train.sh [dataset] [net] [ckpt_to_init] [data_dir] [step]

set -x
set -e

export PYTHONUNBUFFERED='True'

step=$1
ckpt_path=$2
# NET=$2
# data_dir=$4

# For my own experiment usage, just ignore it.
if [ -d '/home/joe' ]; then
    # NET='res50'
    # ckpt_path="experiments/rd_fixconv_i165k_171221/dc_conv_fixed/vg_1.2_train"
    ckpt_path='/home/joe/git/furniture/models/resnet_v2_101.ckpt'
fi

LOG="logs/step_${step}_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ ${step} -lt '2' ]
then
time python ./lib/train_net.py \
    --weights ${ckpt_path} \
    --iters 10000 \
    --set EXP_DIR conv_fixed RESNET.FIXED_BLOCKS 3
fi

if [ ${step} -lt '3' ]
then
NEW_WEIGHTS=output/conv_fixed/ckpt
time python ./lib/train_net.py \
    --weights ${NEW_WEIGHTS} \
    --iters 10000 \
    --set EXP_DIR finetune1 RESNET.FIXED_BLOCKS 2 TRAIN.LEARNING_RATE 0.0005
fi

if [ ${step} -lt '4' ]
then
NEW_WEIGHTS=output/finetune1/ckpt
time python ./lib/train_net.py \
    --weights ${NEW_WEIGHTS} \
    --iters 5000 \
    --set EXP_DIR finetune2 RESNET.FIXED_BLOCKS 1 TRAIN.LEARNING_RATE 0.00025
fi

# if [ ${step} -lt '5' ]
# then
# NEW_WEIGHTS=output/finetune2/ckpt
# time python ./lib/train_net.py \
#     --weights ${NEW_WEIGHTS} \
#     --iters 10000 \
#     --set EXP_DIR finetune3 RESNET.FIXED_BLOCKS 0 TRAIN.LEARNING_RATE 0.0005
# fi

