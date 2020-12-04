"""
main training 
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

## Network
data = np.load(opt.data_path)
label = np.load(opt.label_path)
net = create_model(opt)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
if os.path.exists(opt.checkpoint_path) and opt.pretrained == True:
    checkpoint = torch.load(opt.checkpoint_path)
    load_checkpoint(net, checkpoint)


## Hyper Parameters
num_epochs = opt.num_epochs
batch_size = opt.batch_size
train_loader, val_loader, test_loader = load_EEG_Datasets(data, label, batch_size, is_val=True)


## Visualize 
vis = visdom.Visdom(env='main', port=opt.display_port)
total_start = time.time()

## start training 
print("training on {} ...".format(device))
train_loss, train_acc = [], []
val_loss, val_acc = [], []
for epoch in range(num_epochs):
    epoch_start = time.time()
    net.train()
    loss, acc = [], []
    for i, (X, y) in enumerate(train_loader):
        y_hat = net(X) # batch_size X 2
        loss_fn = criterion(y_hat, y)
        loss.append(loss_fn.item())
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
        acc.append(Accuracy(y_hat, y))

    train_loss.append(sum(loss) / len(loss))
    train_acc.append(sum(acc) / len(acc))
    # print(train_loss)
    
    print("epoch: %d, training loss: %.4f, training accuracy: %.4f, time: %d"%(
        epoch+1, train_loss[epoch], train_acc[epoch], time.time() - epoch_start
    ))

    ## validation
    net.eval()
    loss, acc = [], []
    for i, (X, y) in enumerate(val_loader):
        y_hat = net(X)
        loss_fn = criterion(y_hat, y)
        loss.append(loss_fn.item())
        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
        acc.append(Accuracy(y_hat, y))

    val_loss.append(sum(loss) / len(loss))
    val_acc.append(sum(acc) / len(acc))
    print("validate loss: %.4f, validate accuracy: %.4f"%(train_loss[epoch], train_acc[epoch]))


    ## Visualize
    vis.line(X=[_ for _ in range(epoch+1)], Y=np.column_stack((train_loss, train_acc, val_loss, val_acc)), win='train', \
        opts={
            'title': opt.model + '--train',
            'dash': np.array(['solid', 'solid', 'solid', 'solid']),
            'legend': ['train_loss', 'train_acc', 'val_loss', 'val_acc'],
            'showlegend': True, 
            })


print("training done !")

## save checkpoint
state = {
    'state_dict': net.state_dict(),
}
save_checkpoint(state, opt.checkpoint_path)


## wheather to test
xx, test_loss, test_acc = [], [], []
if opt.test == True:
    print("=="*10, "test begin !", "=="*10)
    print("testing on {} ...".format(device))
    net.eval()
    for epoch in range(num_epochs):
        loss, acc = 0, 0 
        epoch_start = time.time()
        for i, (X, y) in enumerate(test_loader):
            y_hat = net(X) # batch_size X 2
            loss_fn = criterion(y_hat, y)
            loss += loss_fn.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = y_hat.max(1)[1]
            acc += (pred == y).sum().item() / len(pred)

        xx.append(epoch)
        test_loss.append(loss.data)
        test_acc.append(acc)

        vis.line(X=xx,Y=np.column_stack((test_loss, test_acc)), win='test', \
            opts={
                'title': opt.model + '--test',
                'dash': np.array(['solid', 'dash']),
                'legend':['loss', 'acc'],
                'showlegend': True
                })
        print("epoch: %d, testing loss: %.4f, testing accuracy: %.4f, time: %d"%(
            epoch+1, sum(test_loss) / len(test_loss), sum(test_acc) / len(test_acc), time.time() - epoch_start
        ))

    print("=="*10, "testing done !", "=="*10)
    print("testing loss: %.4f, testing accuracy: %.4f, total time: %d"%(
            sum(test_loss) / len(test_loss), sum(test_acc) / len(test_acc), time.time() - total_start    
        ))