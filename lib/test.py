from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from config import cfg
from tqdm import tqdm
from six.moves import cPickle as pickle
import numpy as np
import os
import pdb
from os.path import join as pjoin
import pandas as pd



def test_net(sess, net, imdb, tag, wlabel=True):
    if not os.path.exists(cfg.RESULT_PATH):
        os.makedirs(cfg.RESULT_PATH)

    num = len(imdb.image_index)
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_iter = num // batch_size if not num % batch_size else num // batch_size + 1
    preds = []
    probs = []
    labels = []
    pbar = tqdm(range(num_iter))
    for i in pbar:
        data, i_label = imdb.get_test_batch(batch_size, cfg.TEST.FLIP)
        i_probs, i_preds = net.inference_step(sess, data)
        if wlabel:
            labels += i_label
            i_acc = np.mean(np.array(i_label) == i_preds)
            pbar.set_description('accuracy: {:.3f}'.format(i_acc))
        if not len(preds):
            preds = i_preds
            probs = i_probs
        else:
            preds = np.concatenate((preds, i_preds))
            probs = np.concatenate((probs, i_probs))
    if wlabel:
        accuracy = np.mean(np.array(labels) == preds)
        print("Over all accuracy for {} set is: {}".format(imdb._image_set, accuracy))
    filename = tag + '.pkl'
    filename = pjoin(cfg.RESULT_PATH, filename)

    if not wlabel:
        save_csv(imdb, preds)

    with open(filename, 'wb') as f:
        if wlabel:
            pickle.dump(accuracy, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(np.array(labels), f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(preds, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(probs, f, pickle.HIGHEST_PROTOCOL)

    print("Data saved to {}".format(filename))


def save_csv(imdb, preds):
    file_path = cfg.DATA_PATH + '/sample_submission_randomlabel.csv'
    sx = pd.read_csv(file_path)
    preds = np.asarray(preds, dtype=np.int32)
    sx.loc[sx.id.isin(imdb.image_index), 'predicted'] = preds
    target_path = cfg.DATA_PATH + '/result.csv'
    print("result will save to {}".format(target_path))
    sx.to_csv(target_path, index=False)


if __name__ == '__main__':
    save_csv()
