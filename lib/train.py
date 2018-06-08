from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from config import cfg
from six.moves import cPickle as pickle
import numpy as np
import os
import cv2
import pdb


import glob
import time
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from timer import Timer


class SolverWrapper(object):
    """docstring for SolverWrapper"""

    def __init__(self, sess, network, imdb, imdb_val, output_dir, tb_dir, pretrained_model=None):
        super(SolverWrapper, self).__init__()
        self.net = network
        self.imdb = imdb
        self.imdb_val = imdb_val
        self.output_dir = output_dir
        self.tb_dir = tb_dir
        self.pretrained_model = pretrained_model
        self.tb_valdir = tb_dir + '_val'
        if not os.path.exists(self.tb_valdir):
            os.makedirs(self.tb_valdir)

    def snapshot(self, sess, iters=0):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = "funiture_iter_{}".format(iters) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print("Wrote snapshot to: {:s}.".format(filename))

        return filename

    def from_snapshot(self, sess, filename):
        print("Restoring from {:s}".format(filename))
        self.saver.restore(sess, filename)
        print("Restore.")
        last_snapshot_iter = filename.split("_")[-1].split(".")[0]

        return int(last_snapshot_iter)

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def construct_graph(self, sess):
        with sess.graph.as_default():
            loss = self.net.create_architecture("TRAIN", tag="defautl")

            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)

            print("learning rate {}".format(cfg.TRAIN.LEARNING_RATE))
            self.global_step = tf.Variable(0, trainable=False)
            if cfg.TRAIN.LR_DIY_DECAY:
                learning_rate = cfg.TRAIN.LEARNING_RATE
            else:
                learning_rate = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE,
                                                           self.global_step,
                                                           cfg.TRAIN.EXP_DECAY_STEPS,
                                                           cfg.TRAIN.EXP_DECAY_RATE,
                                                           staircase=True)
            if cfg.TRAIN.OPTIMIZER == 'sgd_m':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM)
            elif cfg.TRAIN.OPTIMIZER == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate)

                # must disable diy decay when using exponentially decay.
                assert cfg.TRAIN.LR_DIY_DECAY == False

            train_op = self.optimizer.minimize(loss, global_step=self.global_step)

            self.saver = tf.train.Saver(max_to_keep=100000)
            self.writer = tf.summary.FileWriter(self.tb_dir, sess.graph)
            self.val_writer = tf.summary.FileWriter(self.tb_valdir)

        return learning_rate, train_op

    def find_previous(self):
        files = os.path.join(self.output_dir, "funiture_iter_*.ckpt.meta")
        files = glob.glob(files)
        files.sort(key=os.path.getmtime)

        sfiles = [ss.replace('.meta', '') for ss in files]

        return len(sfiles), sfiles

    def initialize(self, sess):
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        if self.pretrained_model:
            print('Loading initial model weights from {:s}'.format(self.pretrained_model))
            var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            print('Loaded.')
            # Need to fix the variables before loading, so that the RGB weights are changed to BGR
            # For VGG16 it also changes the convolutional weights fc6 and fc7 to
            # fully connected weights
            if self.net._net_type == "resnet":
                self.net.fix_variables(sess, self.pretrained_model)
                print('Fixed.')
            print("Ckpt path: {}".format(self.pretrained_model))

        return 0

    def train_model(self, sess, max_iters):
        lr, train_op = self.construct_graph(sess)
        lfiles, sfiles = self.find_previous()
        if not lfiles:
            last_snapshot_iter = self.initialize(sess)
        else:
            last_snapshot_iter = self.from_snapshot(sess, str(sfiles[-1]))
            #sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
            #sess.run(tf.assign(self.global_step, 0))

        iters = last_snapshot_iter + 1
        timer = Timer()
        last_summary_time = time.time()
        batch_size = cfg.TRAIN.BATCH_SIZE
        while iters < max_iters + 1:
            timer.tic()

            data, labels = self.imdb.get_minibatch(batch_size)

            now = time.time()

            if iters == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                acc, cls_ce, summary = self.net.train_step_with_summary(sess, data, labels, train_op)
                self.writer.add_summary(summary, float(iters))
                if self.imdb_val:
                    dval, lval = self.imdb_val.get_minibatch(cfg.TRAIN.BATCH_SIZE)
                    acc, cls_ce, summary_val = self.net.val_step_with_summary(sess, dval, lval)
                    self.val_writer.add_summary(summary_val, float(iters))
            else:
                acc, cls_ce = self.net.train_step(sess, data, labels, train_op)

            timer.toc()

            # Display training information
            if iters % (cfg.TRAIN.DISPLAY) == 0:
                if cfg.TRAIN.LR_DIY_DECAY:
                    learning_rate = lr
                else:
                    learning_rate = sess.run(lr)
                print('iters: %d / %d, loss: %.6f\n accuracy: %.6f\n >>> lr: %f' %
                      (iters, max_iters, cls_ce, acc, learning_rate))
                print('speed: {:.3f}s / iters'.format(timer.average_time))

            if np.isnan(cls_ce):
                pdb.set_trace()
                print(self.imdb.image_index[self.imdb._perm[self.imdb._cur - batch_size:self.imdb._cur]])

            # Snapshotting
            if iters % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iters
                ss_path = self.snapshot(sess, iters)

                # Remove the old snapshots if there are too many
                # if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                #     self.remove_snapshot(np_paths, ss_paths)

            iters += 1

        if last_snapshot_iter != iters - 1:
            filename = self.snapshot(sess, iters - 1)

        self.writer.close()
        self.val_writer.close()


def train_net(network, imdb, imdb_val, output_dir, tb_dir, pretrained_model=None, max_iters=10000):

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, imdb, imdb_val, output_dir, tb_dir, pretrained_model=pretrained_model)

        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')
