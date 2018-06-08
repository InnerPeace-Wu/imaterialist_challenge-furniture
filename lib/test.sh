#!/usr/bin/env bash

set -x
set -e

TAG=$1
IMDB=$2
FLIP=$3
CKPT=$4

if [ -d '/home/joe' ]; then
    CKPT='/home/joe/git/furniture/output/NasNet_focal/ckpt'
fi

LOG="logs/test_${IMDB}_${TAG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/test_net.py \
    --ckpt ${CKPT} \
    --imdb ${IMDB} \
    --tag ${TAG} \
    --set TRAIN.BATCH_SIZE 5 NET "nasnet" TRAIN.IMG_WIDTH 299 TRAIN.IMG_HEIGHT 299 TEST.FLIP ${FLIP}
