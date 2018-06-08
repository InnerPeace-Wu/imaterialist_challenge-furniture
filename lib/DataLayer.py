from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
from PIL import Image
import pdb
# from PIL import Image
import numpy as np
from config import cfg


class DataLayer(object):
    """docstring for DataLayer"""

    def __init__(self, image_set):
        super(DataLayer, self).__init__()
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_PATH, image_set)
        self.image_index = self._load_image_set_index()
        self._num_example = len(self.image_index)
        self._get_json_data()
        self._get_cfg_key()
        self._shuffle_inds()

    def _get_json_data(self):
        json_path = os.path.join(cfg.DATA_PATH, "_%s.json" % self._image_set)
        with open(json_path, 'r') as j:
            self._json_data = json.load(j)

    def _shuffle_inds(self):
        self._cur = 0
        if self._image_set == "train":
            self._perm = np.random.permutation(np.arange(len(self.image_index)))
        else:
            self._perm = np.arange(len(self.image_index))

    def _get_cfg_key(self):
        if type(self._image_set) == bytes:
            self._cfg_key = self._image_set.upper().decode('utf-8')

    def get_test_batch(self, batch_size, test_flip=False):
        if self._cur + batch_size > len(self.image_index):
            bs = len(self.image_index) - self._cur
        else:
            bs = batch_size
        inds = self._perm[self._cur: self._cur + bs]
        self._cur += batch_size
        # TODO(innerpeace): stick to training option
        w, h = cfg.TRAIN.IMG_WIDTH, cfg.TRAIN.IMG_HEIGHT
        data = np.zeros([bs, h, w, 3], dtype=np.float32)
        indexes = self.image_index[inds]
        label = []
        for i, idx in enumerate(indexes):
            im_path = self._json_data[idx]["path"]
            if self._image_set != "test":
                label.append(self._json_data[idx]["label"])
            # im = cv2.imread(im_path)
            im = Image.open(im_path)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.asarray(im, dtype=np.float32)
            if test_flip:
                im = im[:, ::-1, :]
            im = cv2.resize(im, (h, w))
            im = 2 * (im / 255.0) - 1.0
            data[i] = im

        return (data, label)

    def get_minibatch(self, batch_size):
        if self._cur + batch_size >= self._num_example:
            self._shuffle_inds()

        inds = self._perm[self._cur:self._cur + batch_size]
        self._cur += batch_size
        w, h = cfg[self._cfg_key].IMG_WIDTH, cfg[self._cfg_key].IMG_HEIGHT
        data = np.zeros([batch_size, h, w, 3], dtype=np.float32)
        label = np.zeros([batch_size, ], dtype=np.int32)
        indexes = self.image_index[inds]
        for i, idx in enumerate(indexes):
            im_path = self._json_data[idx]["path"]
            label[i] = self._json_data[idx]["label"]
            im = cv2.imread(im_path)
            im = im.astype(np.float32, copy=False)
            if self._json_data[idx]["flipped"]:
                im = im[:, ::-1, :]
            im = cv2.resize(im, (h, w))
            ##############################################
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # b, g, r = cv2.split(im)
            # im = cv2.merge([r, g, b])
            ##############################################
            im = 2 * (im / 255.0) - 1.0
            data[i] = im

        return (data, label)

    def _load_image_set_index(self):
        path = os.path.join(cfg.DATA_PATH, "%s.txt" % self._image_set)
        with open(path, 'r') as f:
            image_index = [line.strip() for line in f.readlines()]
        print("Loaded {} set from {}, with totally {} examples".format(self._image_set, path, len(image_index)))
        image_index = np.asarray(image_index)
        if cfg.DEBUG:
            _perm = np.random.permutation(np.arange(len(image_index)))
            image_index = image_index[_perm[:num_test]]

        return image_index


if __name__ == '__main__':

    train = DataLayer("train")
    import time
    t1 = time.time()
    tmp = train.get_minibatch(32)
    print("time: %.2f" % (time.time() - t1))
    from IPython import embed
    embed()
