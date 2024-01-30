'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import sys

import torch.nn as nn
import torch.nn.init as init

module_path = '/PATH/TO/LAYER/FOLDER/HERE/'
sys.path.append(module_path)
print(sys.path)
from MeanAdaptedConv import MeanAdaptedConv
from MeanAdaptedConvReLU import MeanAdaptedConvReLU
from MeanAdaptedLayer import MeanAdaptedLinear
from RotationLayer import RotationLinear
from RotationConv import RotationConv

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        ma = False
        rot = False
        for module in self.modules():
            if isinstance(module, MeanAdaptedConv) or isinstance(module, MeanAdaptedConvReLU):
                print("Found MeanAdaptedConv in self.modules")
                ma = True
                break
            if isinstance(module, RotationConv):
                print("Found RotationConv in self.modules")
                rot = True
                break

        if ma:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                # nn.Linear(512, 512),
                MeanAdaptedLinear(512, 512, gamma=0.99, bias=True, use_var=True, use_mean=True),
                nn.ReLU(True),
                nn.Dropout(),
                # nn.Linear(512, 512),
                MeanAdaptedLinear(512, 512, gamma=0.99, bias=True, use_var=True, use_mean=True),
                nn.ReLU(True),
                # nn.Linear(512, 10),
                MeanAdaptedLinear(512, 10, gamma=0.99, bias=True, use_var=True, use_mean=True),
            )
        elif rot:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                RotationLinear(512, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),
                nn.Dropout(),
                RotationLinear(512, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),
                #RotationLinear(512, 10),
                MeanAdaptedLinear(512, 10, gamma=0.99, bias=True, use_var=True, use_mean=True),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 10),
            )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, MeanAdaptedConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, MeanAdaptedConvReLU):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, RotationConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, ma=False, relu_integrated=False, rot=False):
    if not ma and not rot:
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    elif rot:
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv2d = RotationConv(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                in_channels = v
        return nn.Sequential(*layers)
    elif ma and not relu_integrated:
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    conv2d = MeanAdaptedConv(in_channels, v, kernel_size=3, padding=1, gamma=0.0, bias=False,
                                             use_var=True, use_mean=True)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    conv2d = MeanAdaptedConv(in_channels, v, kernel_size=3, padding=1, gamma=0.99, bias=True,
                                             use_var=True, use_mean=True)
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    else:
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv2d = MeanAdaptedConvReLU(in_channels, v, kernel_size=3, padding=1, gamma=0.99, bias=True,
                                             use_var=True, use_mean=True)
                layers += [conv2d]
                in_channels = v
        return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_rot():
    """VGG 11-layer model (configuration "A") using only rotations"""
    return VGG(make_layers(cfg['A'], rot=True, ma=False))


def vgg11ma():
    """VGG 11-layer model mean adapted (configuration "A")"""
    return VGG(make_layers(cfg['A'], ma=True))


def vgg11ma_bn():
    """VGG 11-layer model mean adapted with BN (configuration "A")"""
    return VGG(make_layers(cfg['A'], ma=True, batch_norm=True))


def vgg11marelu():
    """VGG 11-layer model mean adapted relu (configuration "A")"""
    return VGG(make_layers(cfg['A'], ma=True, relu_integrated=True))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
