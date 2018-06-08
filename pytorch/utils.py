import torch
from torch.autograd import Variable
from tqdm import tqdm
import pdb
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()


class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)


def predict(model, dataloader, test=False):
    all_labels = []
    all_outputs = []
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader))
    for dat in pbar:
        if not test:
            inputs, labels = dat
            all_labels.append(labels)
        else:
            inputs = dat

        inputs = Variable(inputs, volatile=True)
        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)
        all_outputs.append(outputs.data.cpu())

    all_outputs = torch.cat(all_outputs)
    if not test:
        all_labels = torch.cat(all_labels)
    if use_gpu:
        if not test:
            all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs


# def safe_stack_2array(a, b, dim=0):
#     if a is None:
#         return b
#     return torch.stack((a, b), dim=dim)

def safe_stack_2array(acc, a):
    a = a.unsqueeze(-1)
    if acc is None:
        return a
    return torch.cat((acc, a), dim=acc.dim() - 1)


def predict_tta(model, dataloaders, test=False):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader, test)
        prediction = safe_stack_2array(prediction, px)

    return lx, prediction

def find_in_list(a_list, x):
    for i, a in enumerate(a_list):
        if a == x:
            return i
    return -1

def update_key(old_key_list):
    blocks = ["denseblock%d" % i for i in range(1,5)]
    for b in blocks:
        if b in old_key_list:
            idx = find_in_list(old_key_list, b)
            tmp_list = old_key_list[idx:]
            tmp_list.insert(1, "block")
            return ".".join(tmp_list)

    trans = ["transition%d" % i for i in range(1, 4)]
    for t in trans:
        if t in old_key_list:
            idx = find_in_list(old_key_list, t)
            tmp_list = old_key_list[idx:]
            if "conv" not in old_key_list:
                tmp_list[0] = "denseblock" + tmp_list[0][-1]
            return ".".join(tmp_list)

    if "norm5" in old_key_list:
        idx = find_in_list(old_key_list, "norm5")
        tmp_list = old_key_list[idx:]
        tmp_list.insert(0, "denseblock4")
        return ".".join(tmp_list)

    if "conv0" in old_key_list or "norm0" in old_key_list:
        idx = find_in_list(old_key_list, "features")
        tmp_list = old_key_list[idx:]
        return ".".join(tmp_list)

    return None
