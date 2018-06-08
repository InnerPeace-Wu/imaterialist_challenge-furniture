from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg, cfg_from_list
from DataLayer import DataLayer
from ResNet import ResNetv2
from NasNet import NasNet
import argparse
import pprint
import time
import os
import sys
import tensorflow as tf
from test import test_net


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--ckpt', dest='ckpt',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb',
                        help='validation, test',
                        default='validation', type=str)
    parser.add_argument('--tag', dest='tag',
                        help='directory name of saving results',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if cfg.NET == "res101":
        net = ResNetv2(101)
    elif cfg.NET == "nasnet":
        net = NasNet()

    imdb = DataLayer(args.imdb)
    wlabel = args.imdb != "test"

    net.create_architecture("TRAIN", num_classes=129, tag='test')
    # read checkpoint file
    if args.ckpt:
        ckpt = tf.train.get_checkpoint_state(args.ckpt)
    else:
        raise ValueError("NO checkpoint found in {}".format(args.ckpt))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    saver = tf.train.Saver()
    with tf.Session(config=tfconfig) as sess:
        print('Restored from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        test_net(sess, net, imdb, args.tag, wlabel)
