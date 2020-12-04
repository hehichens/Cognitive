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

class EEGNet(nn.Module):
    def __init__(self, opt):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 101), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*2, 2)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x)) # 1 x 16 x 40 x 38
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2) # 1 x 38 x 16 x 40
        
        
        # Layer 2
        x = self.padding1(x) # 1 x 38 x 17(16+0+1) x 73(40+16+17)
        x = F.elu(self.conv2(x)) # 1 x 4 x 16 x 42 
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x) # 1 x 4 x 4 x 11

        
        # Layer 3
        x = self.padding2(x) # 1 x 4 x 11 x 14
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x) # 1 x 4 x 4 x 11
        x = F.dropout(x, 0.25)
        x = self.pooling3(x) # 1 x 4 x 2 x 2
        
        
        # FC Layer
        x = x.view(-1, 4*2*2) # 1 x 16
        x = torch.sigmoid(self.fc1(x))
        return x


if __name__ == "__main__":
    from utils.options import opt
    net = EEGNet(opt)
    x = torch.rand(2, 1, 40, 101) # 1 x 1 x 120 x 64
    print(net(x))