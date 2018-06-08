import json
from pathlib import Path

import os
import pandas as pd
import numpy as np
import pdb
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip

NB_CLASSES = 128
IMAGE_SIZE = 224

wlable = True


class FurnitureDataset(Dataset):
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        if preffix == 'val':
            path = 'validation'
        else:
            path = preffix
        # path = f'data/{path}.json'
        path = '../data/{}.json'.format(path)
        self.transform = transform
        # img_idx = {int(p.name.split('.')[0])
        #            for p in Path(f'tmp/{preffix}').glob('*.jpg')}
        if preffix == "test":
            txt_file = "test.txt"
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    img_idx = [int(line.strip()) for line in f.readlines()]

            #######################################################
            #     img_idx = []
            #     # img_bar = tqdm(Path('../data/%s' % preffix).glob('*.jpg'), total=len(Path))
            #     # for p in Path('../data/%s' % preffix).glob('*.jpg'):
            #     count = 0
            #     for p in Path('../data/%s' % preffix).glob('*.jpg'):
            #         count += 1
            #         if count % 100 == 0:
            #             print(count)
            #         i_path = '../data/%s/' % preffix + p.name
            #         im = Image.open(i_path)
            #         if im.mode == "RGB":
            #             img_idx.append(int(p.name.split('.')[0]))
            #     print("Dumping ids to file: %s" % txt_file)
            #     with open(txt_file, 'w') as f:
            #         for i in img_idx:
            #             f.write("%s\n" % str(i))
            #######################################################
        else:
            img_idx = {int(p.name.split('.')[0].split('_')[0])
                       for p in Path('../data/%s' % preffix).glob('*.jpg')}


            # Training set testing
            # img_idx = [int(p.name.split('.')[0].split('_')[0])
            #            for p in Path('../data/%s' % preffix).glob('*.jpg')]
            # img_idx = np.asarray(img_idx)
            # np.random.shuffle(img_idx)
            # img_idx = img_idx[:5000]
        raw_data = json.load(open(path))
        if 'annotations' in raw_data:
            annotations = raw_data['annotations']
            data = pd.DataFrame(raw_data['annotations'])
        else:
            data = pd.DataFrame(raw_data['images'])
        self.full_data = data
        nb_total = data.shape[0]
        data = data[data.image_id.isin(img_idx)].copy()
        if wlable and preffix != "test":
            lable_dic = {}
            for ann in annotations:
                lable_dic[ann['image_id']] = ann['label_id']
            data['path'] = data.image_id.map(lambda i: "../data/{}/{}_{}.jpg".format(preffix, i, lable_dic[i]))
        else:
            data['path'] = data.image_id.map(lambda i: "../data/{}/{}.jpg".format(preffix, i))
        self.data = data
        print('[+] dataset `{}` loaded {} images from {}'.format(preffix, data.shape[0], nb_total))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['path'])
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.preffix != "test":
            target = row['label_id'] - 1 if 'label_id' in row else -1
            return img, target
        else:
            return img


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])
preprocess_hflip = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    HorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
preprocess_for_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
preprocess_with_augmentation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 30, IMAGE_SIZE + 30)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3,
                           contrast=0.3,
                           saturation=0.3),
    transforms.ToTensor(),
    normalize
])
