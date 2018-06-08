from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from config import cfg
from six.moves import cPickle as pickle
from collections import Counter
from test import save_csv
from DataLayer import DataLayer
import numpy as np
import pdb

wlabel = False
PKLS = ["nasnetf_45k_test", "nasnetf_45k_test_flip", "nasnetf_40k_test", "nasbig2_45k_test"]

# Nasnet test


def main():
    path = cfg.RESULT_PATH
    pkls = PKLS
    preds = []
    probs = []
    res = 0
    for i, f in enumerate(pkls):
        file = path + '/' + f + '.pkl'
        with open(file, 'rb') as fid:
            if wlabel:
                i_acc = pickle.load(fid)
                labels = pickle.load(fid)
            preds.append(pickle.load(fid)[:, None])
            i_p = pickle.load(fid)
        res += 1.0 / len(PKLS) * i_p

    n_preds = np.argmax(res, axis=1)

    if wlabel:
        accuracy = np.mean(n_preds == labels)
        print("accuracy with taking mean of probs: {}".format(accuracy))
        preds_2 = bin_count(np.concatenate(preds, axis=1))
        accuracy_2 = np.mean(preds_2 == labels)
        print("accuracy with predicts voting: {}".format(accuracy_2))
    else:
        imdb = DataLayer('test')
        save_csv(imdb, n_preds)


def bin_count(a):
    '''find out the most frequent one.'''
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    o = np.zeros((a.shape[0],), dtype=np.int32)
    for i in xrange(a.shape[0]):
        o[i] = c_counter(a[i])

    return o


def c_counter(a):
    c = Counter(a)
    re = sorted(c.items(), key=lambda x: x[1], reverse=True)
    return re[0][0]


if __name__ == '__main__':
    main()
