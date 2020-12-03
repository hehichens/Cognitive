"""
simplify code but the net work is simple
edit by hichens
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import sys; sys.path.append("..")
from utils.options import opt
from utils.utils import *
from models.basemodel import CNN_DEAP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = CNN_DEAP().to(device)
data = np.load(opt.data_path)
label = np.load(opt.label_path)
num_epochs = opt.num_epochs

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
train(net, data, label, num_epochs, criterion, optimizer)