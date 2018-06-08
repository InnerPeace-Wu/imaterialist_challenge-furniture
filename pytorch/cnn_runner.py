import argparse

import pdb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from dense_attention import dense_attention201

import os
import models
import utils
from utils import RunningMean, use_gpu
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, NB_CLASSES, preprocess_hflip, preprocess_for_test, preprocess_with_augmentation_big

USE_FOCAL_LOSS = False

BATCH_SIZE = 16
STARTER = 0

EPOCH = 9
OUTPUT_PATH = "./"
_EPOCHS = [12,13,14,17,19,21,22]


def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.densenet201_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def predict(epoch=None, attention=False):
    BATCH_SIZE = 8
    if not epoch:
        epoch = EPOCH
    if attention:
        save_name = "att_"
    else:
        save_name = ""
    pth_path = OUTPUT_PATH + save_name + 'best_val_weight_%s.pth' % epoch
    print("loading %s" % pth_path)
    if not attention:
        model = get_model()
    else:
        print("loading model...")
        model = dense_attention201(pretrained=False, num_classes=128)
        if use_gpu:
            model.cuda()
        print("done.")
    model.load_state_dict(torch.load(pth_path))
    model.eval()
    tta_preprocess = [preprocess_for_test, preprocess_for_test, preprocess_for_test, preprocess, preprocess_hflip]

    ################### TEST VALIDATION SET
    # data_loaders = []
    # for transform in [preprocess]:
    #     test_dataset = FurnitureDataset('validation', transform=transform)
    #     data_loader = DataLoader(dataset=test_dataset, num_workers=0,
    #                              batch_size=BATCH_SIZE,
    #                              shuffle=False)
    #     data_loaders.append(data_loader)

    # lx, px = utils.predict_tta(model, data_loaders)
    # data = {
    #     'lx': lx.cpu(),
    #     'px': px.cpu(),
    # }
    # _, preds = torch.max(px, dim=1)
    # accuracy = torch.mean((preds.view(-1) != lx).float())
    # print("accuracy: {:.5f}".format(accuracy))
    # torch.save(data, save_name + 'val_prediction.pth')
    ################### TEST VALIDATION SET

    data_loaders = []
    print("number of tta: {}".format(len(tta_preprocess)))
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=0,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders, test=True)
    data = {
        #'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data,  save_name + 'test_prediction_e%s.pth' % epoch)


def train(attention=False):
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation)
    train_val_dataset = FurnitureDataset('validation', transform=preprocess_with_augmentation)
    val_dataset = FurnitureDataset('validation', transform=preprocess)

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    train_val_data_loader = DataLoader(dataset=val_dataset, num_workers=8,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=0,
                                        batch_size=BATCH_SIZE // 2,
                                        shuffle=False)


    if USE_FOCAL_LOSS:
        criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    print("loading model...")
    if not attention:
        model = get_model()
        save_name = ""
    else:
        save_name = "att_"
        model = dense_attention201(num_classes=128)
        if use_gpu:
            model.cuda()
        fresh_params = [p['params'] for p in model.fresh_params()]
        nb_learnable_params = 0
        for pp in fresh_params:
            nb_learnable_params += sum(p.numel() for p in pp)
        print('[+] nb learnable params {}'.format(nb_learnable_params))
    print("done.")

    min_loss = float("inf")
    patience = 0

    for epoch in range(STARTER, STARTER+10):
        print('epoch {}'.format(epoch))
        if epoch == 1:
            lr = 0.00002
            model.load_state_dict(torch.load('best_val_weight_0.pth'))
            print("[+] loading best_val_weight_0.pth")
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('best_val_weight.pth'))
            lr = lr / 5
        elif epoch + 1 % 3 == 0:
            ckpt = save_name+ 'best_val_weight_%s.pth' % (epoch - 1)
            if not os.path.exists(ckpt):
                ckpt = save_name+ 'best_val_weight.pth'
            print("loading {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt))
            lr = lr / 2

        if epoch == 0:
            lr = 0.001
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        print('[+] set lr={}'.format(lr))
        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        ### FOR TRAINING VALIDATION SET
        # if epoch - STARTER + 1 % 2 == 0 and epoch - STARTER > 4:
        #     loader =train_val_data_loader
        #     print("[+] trianing with validation set")
        # else:
        #     loader = training_data_loader
        ### FOR TRAINING VALIDATION SET
        loader = training_data_loader
        pbar = tqdm(loader, total=len(loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            target = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)
            loss = criterion(outputs, target)
            if USE_FOCAL_LOSS:
                y_index = torch.LongTensor(np.arange(labels.shape[0])).cpu()
                l_weight = F.softmax(outputs, dim=1).cpu()[y_index, torch.LongTensor(labels)]
                l_weight = l_weight.detach()
                loss = torch.mean(4 * l_weight.cuda() * loss)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != target.data, dtype=torch.float32), batch_size)
            loss.backward()
            optimizer.step()

            pbar.set_description('{:.5f} {:.3f}'.format(running_loss.value, running_score.value))
        print('[+] epoch {} {:.5f} {:.3f}'.format(epoch, running_loss.value, running_score.value))

        torch.save(model.state_dict(), save_name + 'best_val_weight_%s.pth' % epoch)

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print('[+] val {:.5f} {:.3f}'.format(log_loss, accuracy))

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'best_val_weight.pth')
            print('[+] val score improved from {:.5f} to {:.5f}. Saved!'.format(min_loss, log_loss))
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


def multi_predict(attention=False):
    for e in _EPOCHS:
        predict(e, attention)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict', 'multi'])
    parser.add_argument("-attention", dest="attention", action="store_true")
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.attention)
    elif args.mode == 'predict':
        predict(args.attention)
    elif args.mode == 'multi':
        multi_predict(args.attention)
