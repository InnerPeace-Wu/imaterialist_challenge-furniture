import os
import re
import pdb
import torch
from torch import nn
from utils import update_key
import torchvision.models as M
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.autograd import Variable

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(num_input_features))
        # self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _Dense_Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Dense_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _A_Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_A_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _Upsample(nn.Sequential):
    """docstring for _Upsample"""

    def __init__(self, num_input_features, num_output_features, conv=True):
        super(_Upsample, self).__init__()
        if conv:
            self.add_module("norm", nn.BatchNorm2d(num_input_features))
            self.add_module("relu", nn.ReLU(inplace=True))
            self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("upsample", nn.Upsample(scale_factor=2, mode="bilinear"))
        # self.add_module("norm1", nn.BatchNorm2d(num_output_features))


class _ATT_Forward(nn.Sequential):
    """docstring for _ATT_Forward"""

    def __init__(self, num_input_features, num_output_features, transition=True, norm2=False):
        super(_ATT_Forward, self).__init__()
        if transition:
            self.add_module("transition", _A_Transition(num_input_features, num_output_features))
            num_input_features = num_output_features
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=2, padding=1, bias=False))
        if norm2:
            self.add_module('norm2', nn.BatchNorm2d(num_output_features))


class DenseAttention(nn.Module):
    """docstring for DenseAttention"""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, attention_dim_ratio=4):
        super(DenseAttention, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # block1
        self.denseblock1 = nn.Sequential()
        self.denseblock1.add_module("block", _DenseBlock(num_layers=block_config[0], num_input_features=num_init_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate))
        num_features_b1 = num_init_features + block_config[0] * growth_rate
        self.denseblock1.add_module('norm', nn.BatchNorm2d(num_features_b1))
        self.denseblock1.add_module('relu', nn.ReLU(inplace=True))
        self.transition1 = _Transition(num_input_features=num_features_b1, num_output_features=num_features_b1 // 2)
        # latteral 1

        # block 2
        num_features_b2 = num_features_b1 // 2
        self.denseblock2 = nn.Sequential()
        self.denseblock2.add_module("block", _DenseBlock(num_layers=block_config[1], num_input_features=num_features_b2, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate))
        num_features_b2 = num_features_b2 + block_config[1] * growth_rate
        self.denseblock2.add_module('norm', nn.BatchNorm2d(num_features_b2))
        self.denseblock2.add_module('relu', nn.ReLU(inplace=True))
        self.transition2 = _Transition(num_input_features=num_features_b2, num_output_features=num_features_b2 // 2)

        # latteral 2
        # block 3
        num_features_b3 = num_features_b2 // 2
        self.denseblock3 = nn.Sequential()
        self.denseblock3.add_module("block", _DenseBlock(num_layers=block_config[2], num_input_features=num_features_b3, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate))
        num_features_b3 = num_features_b3 + block_config[2] * growth_rate
        self.denseblock3.add_module('norm', nn.BatchNorm2d(num_features_b3))
        self.denseblock3.add_module('relu', nn.ReLU(inplace=True))
        self.transition3 = _Transition(num_input_features=num_features_b3, num_output_features=num_features_b3 // 2)

        # block 4
        num_features_b4 = num_features_b3 // 2
        self.denseblock4 = nn.Sequential()
        self.denseblock4.add_module("block", _DenseBlock(num_layers=block_config[3], num_input_features=num_features_b4, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate))
        num_features_b4 = num_features_b4 + block_config[3] * growth_rate

        # Final batch norm in densenet
        self.denseblock4.add_module("norm5", nn.BatchNorm2d(num_features_b4))
        self.denseblock4.add_module('relu', nn.ReLU(inplace=True))
        # self.norm5 = nn.BatchNorm2d(num_features_b4)

        # TOP DOWN
        lateral_dim = num_features_b1
        self.conv_b4 = nn.Conv2d(num_features_b4, lateral_dim, kernel_size=1, stride=1, bias=False)
        self.conv_b4_cls = nn.Conv2d(num_features_b4, lateral_dim * 4, kernel_size=1, stride=1, bias=False)
        self.conv_b4_3 = _Upsample(lateral_dim, lateral_dim, conv=False)
        self.conv_b3 = nn.Conv2d(num_features_b3, lateral_dim, kernel_size=1, stride=1, bias=False)
        self.conv_b3_2 = _Upsample(2 * lateral_dim, lateral_dim)
        self.conv_b2 = nn.Conv2d(num_features_b2, lateral_dim, kernel_size=1, stride=1, bias=False)
        self.conv_b2_1 = _Upsample(2 * lateral_dim, lateral_dim)

        # ATTENTION forward
        self.att_b1 = _A_Transition(lateral_dim * 2, lateral_dim)
        self.att_b1_2 = _ATT_Forward(lateral_dim, lateral_dim, transition=False)
        self.att_b2 = _A_Transition(lateral_dim * 2, lateral_dim)
        self.att_b2_3 = _ATT_Forward(lateral_dim * 2, lateral_dim)
        self.att_b3 = _A_Transition(lateral_dim * 2, lateral_dim)
        # self.att_b3_4 = _ATT_Forward(lateral_dim * 2, lateral_dim * 4) #, norm2=True)
        self.att_b3_4 = nn.Sequential(OrderedDict([
            ("denseblock5", _DenseBlock(num_layers=8, num_input_features=lateral_dim * 2, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            ("transition", _Dense_Transition(lateral_dim * 2 + 8 * growth_rate, lateral_dim * 2 + 8 * growth_rate)),
            ("denseblock6", _DenseBlock(num_layers=8, num_input_features=lateral_dim * 2 + 8 * growth_rate, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)),
            # ("norm", nn.BatchNorm2d(lateral_dim*2+16*growth_rate)),
            # ("relu", nn.ReLU(inplace=True)),
        ]))

        # final norm
        self.norm6 = nn.BatchNorm2d(lateral_dim * 8)

        self.classifier = nn.Linear(lateral_dim * 8, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _attention(self, x, att):
        attention = F.sigmoid(att)
        return torch.cat([x, x * attention], 1)

    # def _lateral_cat(self, x, y):
        # _, _, H, W = y.size()
        # x_up = F.upsample(x, size=(H, W), mode='bilinear')
        # return torch.cat([y, F.relu(x, inplace=True)], 1)

    def fresh_params(self):
        params = [
        {'params': self.conv_b4.parameters()},
        {'params': self.conv_b4_cls.parameters()},
        {'params': self.conv_b4_3.parameters()},
        {'params': self.conv_b3.parameters()},
        {'params': self.conv_b3_2.parameters()},
        {'params': self.conv_b2.parameters()},
        {'params': self.conv_b2_1.parameters()},
        {'params': self.att_b1.parameters()},
        {'params': self.att_b1_2.parameters()},
        {'params': self.att_b2.parameters()},
        {'params': self.att_b2_3.parameters()},
        {'params': self.att_b3.parameters()},
        {'params': self.att_b3_4.parameters()},
        {'params': self.norm6.parameters()},
        {'params': self.classifier.parameters()},
        ]

        return params

    def forward(self, x):
        # bottom up
        features = self.features(x)
        # pdb.set_trace()
        b1 = self.denseblock1(features)
        t1 = self.transition1(b1)
        b2 = self.denseblock2(t1)
        t2 = self.transition2(b2)
        b3 = self.denseblock3(t2)
        t3 = self.transition3(b3)
        b4 = self.denseblock4(t3)

        # top down
        conv_b4 = self.conv_b4(b4)
        conv_b3 = self.conv_b3(b3)
        conv_b4_up = self.conv_b4_3(conv_b4)
        # cat_b3 = _lateral_cat(conv_b4_up, conv_b3)
        cat_b3 = torch.cat([conv_b3, conv_b4_up], 1)
        conv_b3_2_up = self.conv_b3_2(cat_b3)
        conv_b2 = self.conv_b2(b2)
        cat_b2 = torch.cat([conv_b2, conv_b3_2_up], 1)
        conv_b2_1_up = self.conv_b2_1(cat_b2)

        # attention
        att_b1 = self.att_b1(self._attention(b1, conv_b2_1_up))
        att_b1_2 = self.att_b1_2(att_b1)
        att_b2 = self.att_b2(self._attention(conv_b2, conv_b3_2_up))
        att_b2_3 = self.att_b2_3(torch.cat([att_b2, att_b1_2], 1))
        att_b3 = self.att_b3(self._attention(conv_b3, conv_b4_up))
        att_b3_4 = self.att_b3_4(torch.cat([att_b3, att_b2_3], 1))

        # classify
        conv_b4_cls = self.conv_b4_cls(b4)
        out = torch.cat([conv_b4_cls, att_b3_4], 1)
        out = self.norm6(out)
        out = F.relu(out, inplace=True)
        out = F.avg_pool2d(out, kernel_size=out.size(2), stride=1).view(out.size(0), -1)
        out = self.classifier(out)
        # pdb.set_trace()

        return out

def dense_attention201(pretrained=True, local_ckpt=None, **kwargs):
    model = DenseAttention(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        if local_ckpt:
            assert os.path.exists(local_ckpt)
            state_dict = torch.load(local_ckpt)
        else:
            state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
                key = new_key

            tmp_list = key.split(".")
            if "classifier" in tmp_list:
                del state_dict[key]
                continue
            # if "conv0" in tmp_list or "norm0" in tmp_list:
                # print(key)
                # continue
            new_key = update_key(tmp_list)
            # print(new_key)
            if new_key:
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def main():
    net = DenseAttention(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32))
    for k in net.state_dict().keys():
        print(k)
    # net(Variable(torch.randn(1, 3, 224, 224)))


def old_keys():
    pth = './output/stage_1/best_val_weight_11.pth'
    s_d = torch.load(pth)
    for k in s_d.keys():
        print(k)


if __name__ == '__main__':
    main()
