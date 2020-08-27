import os
import torch
import torch.nn as nn
from torchvision.models.vgg import VGG
import torch.optim as optim
from torchvision import models

DEBUG = False

class FCN8s(nn.Module):

    def __init__(self, encoder_net, n_class):
        super().__init__()
        self.encoder_net = encoder_net
        self.Deconv1 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 2, padding=1, dilation=1,output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.Deconv2 = nn.ConvTranspose2d(512, 256, kernel_size = (3,4), stride = 2, padding=1,dilation=1,output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.Deconv3 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding=1,dilation=1,output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.Deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.Deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=(5,3), stride=2, padding=(0,1), dilation=1,output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(32, n_class+1, kernel_size=1)

    def forward(self,input):
        output_vgg = self.encoder_net(input)
        v5 = output_vgg["x5"] #(n, 512, in.h/32, in.w/32)
        if DEBUG:
            print("v5 : {}".format(v5.shape))
        v4 = output_vgg['x4'] #(n, 512, in.h/16, in.w/16)
        if DEBUG:
            print("v4 : {}".format(v4.shape))
        v3 = output_vgg['x3'] #(n, 256, in.h/8,  in.h/8)
        if DEBUG:
            print("v3 : {}".format(v3.shape))

        score = self.relu(self.Deconv1(v5))  
        if DEBUG:             # size=(N, 512, in.h/16, in.w/16)
            print("Deconv1: {}".format(score.shape))
        score = self.bn1(score + v4)                      # skip connection, size=(N, 512, in.h/16, in.w/16)
        score = self.relu(self.Deconv2(score))  
        if DEBUG: 
            print("Deconv2: {}".format(score.shape))         # size=(N, 256, in.h/8, in.w/8)
        score = self.bn2(score + v3)                      # element-wise add, size=(N, 256, in.h/8, in.w/8)
        score = self.bn3(self.relu(self.Deconv3(score))) 
        if DEBUG:
            print("Deconv3: {}".format(score.shape)) # size=(N, 128, in.h/4, in.w/4)
        score = self.bn4(self.relu(self.Deconv4(score)))
        if DEBUG:
            print("Deconv4 {}".format(score.shape))  # size=(N, 64, in.h/2, in.w/2)
        score = self.bn5(self.relu(self.Deconv5(score)))  # size=(N, 32, in.h, in.w)
        if DEBUG:
            print("Deconv5: {}".format(score.shape))
        score = self.classifier(score) 
        if DEBUG:
            print("out: {}".format(score.shape))                   # size=(N, n_class, in.h/1, in.h/1)
        
       

        return score  # size=(N, n_class, in.h/1, in.w/1)



class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

ranges = {
'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
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


