from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import argparse
import numpy as np
import tensorflow as tf

from config import cfg, get_output_dir, get_output_tb_dir, cfg_from_list
from DataLayer import DataLayer
from train import train_net
from ResNet import ResNetv2
from NasNet import NasNet
import pprint


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Dense Caption network')
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    # TODO: add inception
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('------ called with args: -------')
    pprint.pprint(args)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)

    imdb = DataLayer("train")
    # imdb_val = DataLayer("validation")
    imdb_val = None

    output_dir = get_output_dir()
    print("output will be saved to `{:s}`".format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir()
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    if cfg.NET == "res101":
        net = ResNetv2(101)
    elif cfg.NET == "nasnet":
        net = NasNet()

    if args.weights and not args.weights.endswith('.ckpt'):
        try:
            ckpt = tf.train.get_checkpoint_state(args.weights)
            pretrained_model = ckpt.model_checkpoint_path
        except:
            raise ValueError("NO checkpoint found in {}".format(args.weights))
    else:
        pretrained_model = args.weights

    # if cfg.DEBUG:
    #     pretrained_model = None

    train_net(net, imdb, imdb_val, output_dir, tb_dir, pretrained_model=pretrained_model, max_iters=args.max_iters)


if __name__ == '__main__':
    main()
