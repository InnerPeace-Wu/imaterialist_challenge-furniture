from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from os.path import join as pjoin
import numpy as np
# from distutils import spawn
# run: pip install easydict
from easydict import EasyDict as edict

__C = edict()
# get config by:
#   from lib.config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Whether to use focal loss to train
__C.TRAIN.USE_FOCAL_LOSS = False

# learning rate manully decay
__C.TRAIN.LR_DIY_DECAY = False

# training optimizer: either 'adam' 'sgd_m'
__C.TRAIN.OPTIMIZER = 'adam'

__C.TRAIN.MOMENTUM = 0.9

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 1e-3

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.001

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Weight initializer
__C.TRAIN.WEIGHT_INITIALIZER = 'normal'

# Image size
__C.TRAIN.IMG_WIDTH = 224
__C.TRAIN.IMG_HEIGHT = 224

__C.TRAIN.EXP_DECAY_STEPS = 5000
__C.TRAIN.EXP_DECAY_RATE = 0.8

__C.TRAIN.BATCH_SIZE = 32

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 300

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

__C.TRAIN.DISPLAY = 10

# TEST OPTIONS

__C.TEST = edict()

__C.TEST.FLIP = False

__C.NET = 'res101'

# Global settings
__C.DATA_PATH = '/home/joe/git/furniture/data'

__C.EXP_DIR = 'default'

__C.DEBUG = False

# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# VGG preprocessing tf.slim
__C.PIXEL_MEANS = np.array([[[123.681, 116.78, 103.94]]])

#
# ResNet options
#

__C.RESNET = edict()

__C.RESNET.FIXED_BLOCKS = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(pjoin(osp.dirname(__file__), '..'))

__C.RESULT_PATH = pjoin(__C.ROOT_DIR, 'result')


def get_output_dir():
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, 'ckpt'))
    # if weights_filename is not None:
    #     outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir():
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, 'tb'))
    # if weights_filename is None:
    #     weights_filename = 'default'
    # outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
