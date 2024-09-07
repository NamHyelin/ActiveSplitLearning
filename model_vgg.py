import math

import torch, torchvision
import os, sys
import torch.nn as nn
import numpy as np
import torchsummary
from torchvision import models
from FLOPs_counter import print_model_parm_flops


'''source : https://github.com/gratus907/Pytorch-Cifar10/blob/main/models/VGGNet.py'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = {
    'vgg_client': [64, 'M', 128, 'M', 256],  #from vggtiny
    'vgg_server': [256, 'M', 512, 512], #from vggtiny
    'vgg_tinyserver': ['M',  512], #from vggtiny
    'vgg_tinyserver_student': ['M',  256, 'M', 512], #from vggtiny
    'vggtiny': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

cfg_tiny = {
    'vgg_tinyserver_0': ['M',  256],
    'vgg_tinyserver_1': ['M',  256, 128],
    'vgg_tinyserver_3': ['M',  512, 256],
    'vgg_tinyserver_4': ['M',  512, 512],
}

def make_layers(cfg, in_channels = 3, stride = 1):
    layers = []
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, x, kernel_size=(3, 3), stride=stride, padding=1)
            layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
            in_channels = x
    return nn.Sequential(*layers)




class VGGNet(nn.Module):
    def __init__(self, config='vggtiny'):
        super().__init__()
        self.name = config
        self.feature_layers = make_layers(cfg[config])
        self.avgpool = nn.AdaptiveAvgPool2d((2)) # new!
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        # out = out.view(-1, 2048)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out







class VGGNet_client(nn.Module):
    def __init__(self, config='vgg_client'):
        super().__init__()
        self.name = config
        self.feature_layers = make_layers(cfg[config])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)

        return out








class VGGNet_server(nn.Module):
    def __init__(self, config='vgg_server'):
        super().__init__()
        self.name = config
        self.feature_layers = make_layers(cfg[config], in_channels=256)
        self.avgpool = nn.AdaptiveAvgPool2d((2)) # new!
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        # out = out.view(-1, 2048)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out







class VGGNet_tinyserver(nn.Module):
    def __init__(self, config='vgg_tinyserver'):
        super().__init__()
        self.feature_layers = make_layers(cfg[config], in_channels=256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1)) # new!
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),  #원래Dropout
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out



class VGGNet_tinyserver_0(nn.Module):
    def __init__(self, config='vgg_tinyserver_0'):
        super().__init__()
        self.feature_layers = make_layers(cfg_tiny[config], in_channels=256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1)) # new!
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),  #원래Dropout
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out



class VGGNet_tinyserver_1(nn.Module):
    def __init__(self, config='vgg_tinyserver_1'):
        super().__init__()
        self.feature_layers = make_layers(cfg_tiny[config], in_channels=256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1)) # new!
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),  #원래Dropout
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out




class VGGNet_tinyserver_3(nn.Module):
    def __init__(self, config='vgg_tinyserver_3'):
        super().__init__()
        self.feature_layers = make_layers(cfg_tiny[config], in_channels=256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1)) # new!
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),  #원래Dropout
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out



class VGGNet_tinyserver_4(nn.Module):
    def __init__(self, config='vgg_tinyserver_4'):
        super().__init__()
        self.feature_layers = make_layers(cfg_tiny[config], in_channels=256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1)) # new!
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),  #원래Dropout
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out



class VGGNet_tinyserver_student(nn.Module):
    def __init__(self, config='vgg_tinyserver_student'):
        super().__init__()
        self.feature_layers = make_layers(cfg[config], in_channels=256)
        self.avgpool = nn.AdaptiveAvgPool2d((2)) # new!
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.feature_layers(x)
        out = self.avgpool(out) #new!
        out= torch.flatten(out,1) #new!
        out = self.classifier(out)
        return out










# s= VGGNet()
# s=s.to(device)
# input= torch.ones((128, 3, 32,32)).to(device)
# o = s(input)


s=VGGNet_tinyserver()
torchsummary.summary(s, (256, 8, 8), device='cpu')  #input 3,32,32  #resnet 64,8,8  #vgg 256, 8, 8
input = torch.randn(128, 256, 8, 8)
print_model_parm_flops(s, input, detail=True)
print('\n')


s= VGGNet_client()
torchsummary.summary(s, (3,32,32), device='cpu')
input = torch.randn(128, 3,32,32)
print_model_parm_flops(s, input, detail=True)
print('\n')


s= VGGNet_server()
torchsummary.summary(s, (256, 8, 8), device='cpu')
input = torch.randn(128, 256, 8, 8)
print_model_parm_flops(s, input, detail=True)

s= VGGNet_tinyserver_student()
torchsummary.summary(s, (256, 8, 8), device='cpu')


# a = 0
# for i in range(len(list(s.named_parameters()))):
#     if 'weight' in list(s.named_parameters())[i][0]:
#         a += np.prod(list(list(s.named_parameters())[i][1].shape))
# print('parameter number: ',a)




