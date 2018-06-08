from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import copy
from os.path import join as pjoin
from six.moves import xrange
from tqdm import tqdm
import numpy as np
import cv2


DEBUG = False
FLIPPE = True
DATA_PATH = '/home/joe/git/furniture/data'
SPLITS = ['train', 'test', 'validation']


def process_dataset(split_name):
    # read json data
    print("Dataset about to process: {}".format(split_name))
    json_file = pjoin(DATA_PATH, split_name + '.json')

    if split_name in ["train", "validation"]:
        with open(json_file, 'r') as j:
            anns = json.load(j)['annotations']
    else:
        with open(json_file, 'r') as j:
            anns = json.load(j)['images']

    if DEBUG:
        anns = anns[:347]
    # store valid image ids
    ids = []
    images = {}
    filter_out = []
    MARGIN = (int(str(len(anns))[:2]) + 1) * 10**(len(str(len(anns))) - 2)
    print("Total number of images: {}, and set MARGIN of flipping id as : {}".format(len(anns), MARGIN))
    for a in tqdm(xrange(len(anns)), desc="%s" % split_name):
        tmp = {}
        id_ = anns[a]["image_id"]
        if split_name == "test":
            path = pjoin(DATA_PATH, split_name, "{}.jpg".format(id_))
            FLIPPE = False
        else:
            path = pjoin(DATA_PATH, split_name, "{}_{}.jpg".format(id_, label))
            label = anns[a]["label_id"]
        # im = cv2.imread(path)
        if os.path.exists(path):  # and isinstance(im, np.ndarray):
            ids.append(id_)
            if split_name in ["train", "validation"]:
                tmp["label"] = label
            tmp["path"] = path
            tmp["flipped"] = False
            images[id_] = tmp
            if FLIPPE and split_name == "train":
                ttmp = copy.deepcopy(tmp)
                ttmp["flipped"] = True
                ids.append(id_ + MARGIN)
                images[id_ + MARGIN] = ttmp
        else:
            filter_out.append(id_)
    print("Number of filter-out images: {}".format(len(filter_out)))
    print("Id of filter_out images: {}".format(filter_out))

    txt_file = "%s/%s.txt" % (DATA_PATH, split_name)
    print("Dumping ids to file: %s" % txt_file)
    with open(txt_file, 'wb') as f:
        for i in ids:
            f.write("%s\n" % i)

    _json_file = "%s/_%s.json" % (DATA_PATH, split_name)
    print("Dumping processed data to file: %s" % _json_file)
    with open(_json_file, 'wb') as _j:
        json.dump(images, _j)


if __name__ == "__main__":
    process_dataset('test')
