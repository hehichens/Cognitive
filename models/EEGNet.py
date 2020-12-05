"""
EEGNet
edit by hichens
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

import sys; sys.path.append("..")
from utils.options import opt 

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class EEGNet(nn.Module):
    def __init__(self, opt):
        super(EEGNet, self).__init__()
        F = [8, 16]
        T = 101
        D = 2
        C = opt.num_channel
        hidden_size = 128
        
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F[0], kernel_size=(1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(F[0], False)
        self.pooling1 = nn.MaxPool2d(kernel_size=(1, 4))

        
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=F[0], out_channels=D*F[0], kernel_size=(C, 1), groups=F[0])
        self.batchnorm2 = nn.BatchNorm2d(D*F[0], False)
        self.pooling2 = nn.MaxPool2d(kernel_size=(1, 8))
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        # self.conv3 = nn.Conv2d(in_channels=D*F[0], out_channels=F[1], kernel_size=(1, 16), groups=1, padding=8)
        self.conv3 = SeparableConv2d(in_channels=D*F[0], out_channels=F[1], kernel_size=(1, 16), padding=(0, 8))
        self.batchnorm3 = nn.BatchNorm2d(F[1], False)
        self.pooling3 = nn.MaxPool2d((1, 16))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.flatten = nn.Flatten()
        size = self.get_size()
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size[1], opt.num_class)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size[1], hidden_size),
            nn.Dropout(opt.dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, opt.num_class),
            nn.ReLU()
        )
        
    def get_feature(self, x):
         # Layer 1
        x = F.relu(self.conv1(x)) # batch_size x 16 x 40 x 101
        x = self.batchnorm1(x)

        if not opt.small:
            x = self.pooling1(x)
            x = F.dropout(x, opt.dropout_rate)
        
        
        # Layer 2
        x = self.conv2(x) # batch_size x 16 x 1 x 102
        x =  F.relu(self.batchnorm2(x))
        if not opt.small:
            x = self.pooling2(x) 
            x = F.dropout(x, opt.dropout_rate)

        
        # Layer 3
        x = F.relu(self.conv3(x)) 
        x = self.batchnorm3(x) 
        if not opt.small:
            x = self.pooling3(x) 
            x = F.dropout(x, opt.dropout_rate)

        return x


    def forward(self, x):
        # FC Layer
        x = self.get_feature(x)
        print(x.shape)
        sys.exit(0)
        x = torch.sigmoid(self.fc(x))
        return x


    def get_size(self):
        x = torch.rand(2, 1, opt.num_channel, opt.num_dim)
        x = self.get_feature(x)
        x = self.flatten(x)
        return x.size()


if __name__ == "__main__":
    from utils.options import opt
    net = EEGNet(opt)
    x = torch.rand(2, 1, opt.num_channel, opt.num_dim) # 1 x 1 x 120 x 64
    print(net(x))