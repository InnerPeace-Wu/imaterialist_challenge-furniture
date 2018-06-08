from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from network import Network
from config import cfg
from nasnet import nasnet_large_arg_scope, build_nasnet_large


class NasNet(Network):
    def __init__(self):
        super(NasNet, self).__init__()
        self._net_type = 'nasnet'

    def _image_to_head(self, is_training, reuse=None):
        with slim.arg_scope(nasnet_large_arg_scope()):
            pool, _ = build_nasnet_large(self._input, 0, is_training=is_training)

        return pool

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            # if v.name == (self._resnet_scope + '/conv1/weights:0'):
            #     self._variables_to_fix[v.name] = v
            #     continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore
