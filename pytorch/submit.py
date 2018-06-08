import torch
import json
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
# from six.moves import cPickle as pickle
import pickle
import numpy as np
from scipy import stats as scistates
import pandas as pd
from misc import FurnitureDataset, preprocess
from collections import Counter

import pdb

DIRS = ["tmp/stage1/", "tmp/stage2/", "tmp/stage6/"]
EPOCHS = [[9,10, 11], [9, 8, 7], [10,9,8,7]]
NAME = 'test_prediction_e%s.pth'


def main():
    test_dataset = FurnitureDataset('test', transform=preprocess)

    # pth = DIRS[2] + 'test_prediction_e6.pth'
    pth = 'test_prediction_e10.pth'
    print("loading {}".format(pth))
    test_pred = torch.load(pth)
    test_prob = F.softmax(Variable(test_pred['px']), dim=1).data.numpy()
    test_prob = test_prob.mean(axis=2)

    test_predicted = np.argmax(test_prob, axis=1)
    test_predicted += 1
    result = test_predicted

    sx = pd.read_csv('../data/sample_submission_randomlabel.csv')
    sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = result
    sx.to_csv(pth.split('.')[0] + '.csv', index=False)


def ensemble():
    test_dataset = FurnitureDataset('test', transform=preprocess)
    probs = []
    # for e in EPOCHS:
    for d, eps in zip(DIRS, EPOCHS):
        for e in eps:
            pth = d + NAME % e
            test_pred = torch.load(pth)
            test_prob = F.softmax(Variable(test_pred['px']), dim=1).data.numpy()
            if len(probs) == 0:
                probs = test_prob
            else:
                probs = np.concatenate((probs, test_prob), axis=-1)

    den_preds = np.argmax(probs, axis=1) + 1
    probs = probs.mean(axis=2)
    # import pdb
    # pdb.set_trace()
    # probs = 0.851 * probs[:, :, :21].mean(axis=2) + 0.863 * probs[:, :, 21:36].mean(axis=2) + 0.855 * probs[:, :, 36:].mean(axis=2)
    nas_probs, nas_preds = read_nasnet()
    en_preds = np.concatenate([den_preds, nas_preds],axis=1)
    probs += nas_probs
    # probs = np.concatenate([probs, nas_probs], axis=2)
    # probs = scistates.gmean(probs, axis=2)
    # probs = 0.85 * probs + 0.86 * nas_probs
    probs = calibrate_probs(probs)
    preds = np.argmax(probs, axis=1)
    preds += 1

    # preds = bin_count(en_preds)

    sx = pd.read_csv('../data/sample_submission_randomlabel.csv')
    sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = preds
    sx.to_csv('ensemble.csv', index=False)


def read_nasnet():
    res_path = '../result'
    # pkls = ["nasbig2_45k_test", "nasbig2_40k_test", "nasbig2_35k_test"]
    accs = [0.8616, 0.8577, 0.8544]  # , 0.8552]
    pkls = ["nasnetf_45k_test", "nasnetf_45k_test_flip", "nasnetf_40k_test"]  #, "nasbig2_45k_test", "nasbig2_40k_test", "nasbig2_35k_test"]
    pkls += ["nasf_%dk_test" % i for i in range(70,100,5)]
    res = 0
    for i, f in enumerate(pkls):
        file = res_path + '/' + f + '.pkl'
        with open(file, 'rb') as fid:
            pickle.load(fid, encoding='iso-8859-1')
            i_p = pickle.load(fid, encoding='iso-8859-1')
        res += 1.0 / len(pkls) * i_p
        tmp = np.argmax(i_p, axis=1)[:, None]
        # for gmean testing
        # i_p = i_p[:,:,None]
        if i == 0:
            # res = i_p
            raw_preds = tmp
        else:
            # res = np.concatenate([res, i_p], axis=2)
            raw_preds = np.concatenate([raw_preds, tmp], axis=1)

    return res[:, 1:], raw_preds

def bin_count(a):
    '''find out the most frequent one.'''
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    o = np.zeros((a.shape[0],), dtype=np.int32)
    for i in range(a.shape[0]):
        o[i] = c_counter(a[i])

    return o


def c_counter(a):
    c = Counter(a)
    re = sorted(c.items(), key=lambda x: x[1], reverse=True)
    return re[0][0]

# borrowed from https://www.kaggle.com/dowakin/probability-calibration-0-005-to-lb
def calibrate(prior_y0_train, prior_y0_test,
              prior_y1_train, prior_y1_test,
              predicted_prob_y0):
    predicted_prob_y1 = (1 - predicted_prob_y0)

    p_y0 = prior_y0_test * (predicted_prob_y0 / prior_y0_train)
    p_y1 = prior_y1_test * (predicted_prob_y1 / prior_y1_train)
    return p_y0 / (p_y0 + p_y1)  # normalization


def calibrate_probs(prob):
    prior_y0_test = 1/128
    prior_y1_test = 1 - prior_y0_test
    train_json  = json.load(open('../data/train.json'))
    train_df = pd.DataFrame(train_json['annotations'])
    calibrated_prob = np.zeros_like(prob)
    nb_train = train_df.shape[0]
    for class_ in range(128): # enumerate all classes
        prior_y0_train = ((train_df.label_id - 1) == class_).mean()
        prior_y1_train = 1 - prior_y0_train

        for i in range(prob.shape[0]): # enumerate every probability for a class
            predicted_prob_y0 = prob[i, class_]
            calibrated_prob_y0 = calibrate(
                prior_y0_train, prior_y0_test,
                prior_y1_train, prior_y1_test,
                predicted_prob_y0)
            calibrated_prob[i, class_] = calibrated_prob_y0
    return calibrated_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['single', 'ensemble'])
    args = parser.parse_args()
    if args.mode == 'single':
        main()
    elif args.mode == 'ensemble':
        ensemble()
