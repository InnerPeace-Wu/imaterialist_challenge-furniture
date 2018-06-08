#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED='True'

LOG="./logs/preprocess.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./lib/preprocess.py
