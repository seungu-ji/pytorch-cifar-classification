"""
Very Deep Convolutional Networks for Large-Scale Image Recognition
arxiv link: https://arxiv.org/abs/1409.1556

VGG-A: 11 weight layers
VGG-A-LRN: 11 weight layers
VGG-B: 13 weight layers
VGG-C: 16 weight layers
VGG-D: 16 weight layers
VGG-E: 19 weight laeyrs
"""

import torch
import torch.nn as nn

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGNet(nn.Module):
    def __init__(self, vgg_type):
        super(VGGNet, self).__init__()
        
        self.layers = self.make_layers(cfg[vgg_type])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            # Maxpool
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)

def vgg11(**kwargs):
    net = VGGNet(vgg_type='A')
    return net

def vgg13(**kwargs):
    net = VGGNet(vgg_type='B')
    return net

def vgg16(**kwargs):
    net = VGGNet(vgg_type='C')
    return net

def vgg19(**kwargs):
    net = VGGNet(vgg_type='E')
    return net


def test():
    #net = VGGNet('E')
    net = vgg19()

    x = torch.randn(2,3,32,32)
    y = net(x)

    return y.size()

print(test())