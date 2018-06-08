from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import cv2
from config import cfg
from DataLayer import DataLayer


def draw_image(image, probs, labels):
    pixel_mean = cfg.PIXEL_MEANS
    num = min(image.shape[0], 8)
    preds = np.argmax(probs, axis=1)
    w, h = cfg.TRAIN.IMG_WIDTH, cfg.TRAIN.IMG_HEIGHT
    out = np.zeros([num, h, w, 3], dtype=np.uint8)
    for i in xrange(num):
        img = image[i]
        img = (img + 1.0) / 2.0 * 255.0
        img = np.asarray(img, dtype=np.uint8)
        string = "PD: %s/%.2f, GT: %s" % (preds[i], probs[i][preds[i]], labels[i])
        cv2.putText(img, string, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        out[i] = img

    return out


def main():
    dl = DataLayer("train")
    image, labels = dl.get_minibatch(8)
    for im in image:
        im = np.asarray(im, dtype=np.uint8)
        # r, g, b = cv2.split(im)
        # im = cv2.merge([b, g, r])
        cv2.imshow("image", im)
        cv2.waitKey(1000)
    out = draw_image(image, labels[:, None], labels)
    for im in out:
        cv2.imshow("image", im)
        cv2.waitKey(1000)


if __name__ == '__main__':
    main()
