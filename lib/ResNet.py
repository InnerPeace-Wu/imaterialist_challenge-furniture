from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim import losses
# from tensorflow.contrib.slim import arg_scope
# from tensorflow.contrib.slim.python.slim.nets import resnet_utils
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1
# from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_block
import numpy as np

from network import Network
from config import cfg
from resnet_v2 import resnet_v2, resnet_v2_block, resnet_v2_101
import resnet_utils
from resnet_utils import resnet_arg_scope


class ResNetv2(Network):
    """docstring for ResNetv2"""

    def __init__(self, num_layers=101):
        super(ResNetv2, self).__init__()
        self._net_type = 'resnet'
        self._num_layers = num_layers
        self._resnet_scope = 'resnet_v2_%d' % num_layers
        self._decide_blocks()

    # Do the first few layers manually, because 'SAME' padding can behave inconsistently
    # for images of different sizes: sometimes 0, sometimes 1
    def _build_base(self):
        with tf.variable_scope(self._resnet_scope):
            net = resnet_utils.conv2d_same(self._input, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def _decide_blocks(self):
        if self._num_layers == 101:
            self._blocks = [
                resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
                resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
                resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
                resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
            ]
        else:
            raise NotImplementedError

    def _resnet_v2(self, is_training):
        with slim.arg_scope(resnet_arg_scope()):
            pool, _ = resnet_v2_101(self._image, is_training=is_training)

        return pool

    def _image_to_head(self, is_training, reuse=None):
        fix = cfg.RESNET.FIXED_BLOCKS
        assert (0 <= fix <= 3)
        # Now the base is always fixed during training
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv = self._build_base()

        print("Fixing %s blocks." % cfg.RESNET.FIXED_BLOCKS)
        with slim.arg_scope(resnet_arg_scope()):
            c1, _ = resnet_v2(net_conv,
                              [self._blocks[0]],
                              is_training=0 >= fix and is_training,
                              global_pool=False,
                              include_root_block=False,
                              reuse=reuse,
                              scope=self._resnet_scope)
            if 0 >= fix and is_training:
                print("training block 1")
            c2, _ = resnet_v2(c1,
                              [self._blocks[1]],
                              is_training=1 >= fix and is_training,
                              global_pool=False,
                              include_root_block=False,
                              reuse=reuse,
                              scope=self._resnet_scope)
            if 1 >= fix and is_training:
                print("training block 2")
            c3, _ = resnet_v2(c2,
                              [self._blocks[2]],
                              is_training=2 >= fix and is_training,
                              global_pool=False,
                              include_root_block=False,
                              reuse=reuse,
                              scope=self._resnet_scope)
            if 2 >= fix and is_training:
                print("training block 3")
            c4, _ = resnet_v2(c3,
                              [self._blocks[3]],
                              is_training=is_training,
                              global_pool=False,
                              include_root_block=False,
                              postnorm=True,
                              reuse=reuse,
                              scope=self._resnet_scope)

            self._layers["c1"] = c1
            self._layers["c2"] = c2
            self._layers["c3"] = c3
            self._layers["c4"] = c4

        return c4

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._resnet_scope + '/conv1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix Resnet V2 layers..')
        with tf.variable_scope('Fix_Resnet_V2') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
