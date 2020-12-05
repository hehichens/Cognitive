"""
test accuracy 
edit by Hichens

Example:
    >>>python main.py --model TSception 
for more option, please see the utils.options for more paramters

"""

import os
import torch
import time
import sys; sys.path.append("..")
import visdom
import numpy as np

from utils.utils import *
from utils.options import opt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    ## Network
    data = np.load(opt.data_path)
    label = np.load(opt.label_path)
    net = create_model(opt).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    if opt.best == True:
        checkpoint_path = os.path.join(opt.checkpoint_dir, opt.model+'_best.pth')
    else:
        checkpoint_path = os.path.join(opt.checkpoint_dir, opt.model+'.pth')
    checkpoint = torch.load(checkpoint_path)
    load_checkpoint(net, checkpoint)


    ## Hyper Parameters
    batch_size = opt.batch_size
    train_loader, val_loader, test_loader = load_EEG_Datasets(data, label, batch_size, is_val=True)


    print("test begin !")
    print("testing on {} ...".format(device))
    net.eval()
    loss, acc = [], []
    for i, (X, y) in enumerate(test_loader):
        y_hat = net(X)
        loss_fn = criterion(y_hat, y)
        loss.append(loss_fn.item())
        acc.append(Accuracy(y_hat, y))

    print("testing loss: %.4f, testing accuracy: %.4f"%(sum(loss) / len(loss), sum(acc) / len(acc)))
