from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
# import tensorflow.contrib.rnn as rnn

from vis import draw_image
import numpy as np
from config import cfg
import pdb

TRY_OFFICIAL = False


class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._layers = {}
        self._train_summaries = []
        self._variables_to_fix = {}
        self._image = tf.placeholder(tf.float32, shape=[None, cfg.TRAIN.IMG_HEIGHT, cfg.TRAIN.IMG_WIDTH, 3])
        self._labels = tf.placeholder(tf.int32, shape=[None, ])
        self._preprocess()

    def _preprocess(self):
        _image = self._image
        #################################################
        #_image = tf.scalar_mul((1.0 / 255), self._image)
        #_image = tf.multiply(_image, 1.0 / 255)
        #_image = tf.subtract(_image, 0.5)
        #_image = tf.multiply(_image, 2.0)
        #################################################

        self._input = _image

    def _add_image_summary(self):
        image = tf.py_func(draw_image, [self._image, self._predictions['probs'], self._labels], tf.uint8, name='image_summary')

        return tf.summary.image("Image", image)

    def _resnet_v2(self, is_training):
        raise NotImplementedError

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def create_architecture(self, mode, num_classes=129, tag=None):
        self._tag = tag
        self._num_classes = num_classes
        training = mode == "TRAIN"
        testing = mode = "TEST"

        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        weights_initializer = tf.contrib.layers.xavier_initializer()
        # slim.variance_scaling_initializer()

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       # biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            if self._net_type == 'resnet':
                if TRY_OFFICIAL:
                    pool5 = self._resnet_v2(is_training=training)
                else:
                    c4 = self._image_to_head(training)
                    pool5 = tf.reduce_mean(c4, [1, 2], name='pool5', keep_dims=True)
            elif self._net_type == 'nasnet':
                pool5 = self._image_to_head(is_training=training)
                pool5 = tf.expand_dims(pool5, 1)
                pool5 = tf.expand_dims(pool5, 1)
            logits = slim.conv2d(pool5, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            probs = slim.softmax(logits, scope='probs')
            prediction = tf.argmax(logits, 1, output_type=tf.int32)
            correct_prediction = tf.equal(prediction, self._labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._predictions['logits'] = logits
        self._predictions['accuracy'] = accuracy
        self._predictions['probs'] = probs
        self._predictions['prediction'] = prediction

        if training:
            loss = self._add_losses()
            val_summaries = []
            tf.summary.scalar("Accuracy", accuracy)
            tf.summary.histogram("logits", logits)
            tf.summary.scalar("Loss", self._losses['cls_ce'])
            val_summaries.append(tf.summary.scalar("Accuracy", accuracy))
            val_summaries.append(tf.summary.scalar("Loss", self._losses['cls_ce']))
            val_summaries.append(self._add_image_summary())

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

            return loss

    def _add_losses(self):
        with tf.variable_scope("LOSS_" + self._tag) as scope:
            logits = tf.reshape(self._predictions['logits'], [-1, self._num_classes])
            if cfg.TRAIN.USE_FOCAL_LOSS:
                probs = self._predictions['probs']
                probs = tf.stop_gradient(probs)
                probs = tf.gather_nd(probs, tf.stack((tf.range(cfg.TRAIN.BATCH_SIZE), self._labels), axis=1))
                # probs = tf.square(1 - probs)
                probs = 1 - probs
                cls_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._labels)
                cls_ce = tf.reduce_mean(probs * cls_ce)
            else:
                cls_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._labels))

            self._losses['cls_ce'] = cls_ce

        return cls_ce

    def train_step(self, sess, data, labels, train_op):
        feed_dict = {self._image: data, self._labels: labels}
        accuracy, loss, _ = sess.run([self._predictions['accuracy'],
            self._losses['cls_ce'], train_op], feed_dict=feed_dict)

        return accuracy, loss

    def inference_step(self, sess, data):
        probs, prediction = sess.run([self._predictions['probs'],
                                      self._predictions['prediction']],
                                      feed_dict={self._image: data})

        return probs, prediction

    def train_step_with_summary(self, sess, data, labels, train_op):
        feed_dict = {self._image: data, self._labels: labels}
        summary_op = self._summary_op
        accuracy, loss, summary, _ = sess.run([self._predictions['accuracy'],
                                               self._losses['cls_ce'],
                                               summary_op, train_op],
                                              feed_dict=feed_dict)

        return accuracy, loss, summary

    def val_step_with_summary(self, sess, data, labels):
        feed_dict = {self._image: data, self._labels: labels}
        summary_op = self._summary_op_val
        accuracy, loss, summary = sess.run([self._predictions['accuracy'],
                                            self._losses['cls_ce'],
                                            summary_op],
                                           feed_dict=feed_dict)

        return accuracy, loss, summary
