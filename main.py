"""
main training 
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
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


## Network
data = np.load(opt.train_data_path)
label = np.load(opt.train_label_path)
net = create_model(opt).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, \
    weight_decay=opt.weight_decay)

checkpoint_path = os.path.join(opt.checkpoint_dir, opt.model+'.pth')
best_checkpoint_path = os.path.join(opt.checkpoint_dir, opt.model+'_best.pth')
if os.path.exists(checkpoint_path) and opt.pretrained == True:
    checkpoint = torch.load(checkpoint_path)
    load_checkpoint(net, checkpoint)


## Hyper Parameters
num_epochs = opt.num_epochs
batch_size = opt.batch_size
train_loader, val_loader = load_EEG_Datasets(data, label, batch_size, is_val=True)


## Visualize 
vis = visdom.Visdom(env='main', port=opt.display_port)


## Train
print("training on {} ...".format(device))
train_loss, train_acc = [], []
val_loss, val_acc = [], []
best_train_acc, best_val_acc, patient = 0, 0, 0
for epoch in range(num_epochs):
    epoch_start = time.time()
    net.train()
    loss, acc = [], []
    for i, (X, y) in enumerate(train_loader):
        y_hat = net(X) # batch_size X 2
        loss_fn = criterion(y_hat, y)
        if opt.normalized:
            loss_r = regulization(net)
            loss_fn += loss_r
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


    ## early stop
    if best_train_acc < train_acc[epoch]:
        best_train_acc = train_acc[epoch]
    elif train_acc[epoch] < best_train_acc * 0.9:
        patient += 1
    if patient > opt.patient:
        print("=> early stop!")
        break


    ## Validation
    net.eval()
    loss, acc = [], []
    for i, (X, y) in enumerate(val_loader):
        y_hat = net(X)
        loss_fn = criterion(y_hat, y)
        
        loss.append(loss_fn.item())
        acc.append(Accuracy(y_hat, y))
    val_loss.append(sum(loss) / len(loss))
    val_acc.append(sum(acc) / len(acc))
    print("validate loss: %.4f, validate accuracy: %.4f"%(val_loss[epoch], val_acc[epoch]))


    ## Save the best state 
    if best_val_acc < val_acc[epoch]:
        best_val_acc = val_acc[epoch]
        state = {
            'state_dict': net.state_dict(),
        }
        save_checkpoint(state, best_checkpoint_path)


    ## Visualize
    vis.line(X=[_ for _ in range(epoch+1)], Y=np.column_stack((train_loss, train_acc, val_loss, val_acc)), win='train', \
        opts={
            'title': opt.model + '--train',
            'dash': np.array(['solid', 'solid', 'solid', 'solid']),
            'legend': ['train_loss', 'train_acc', 'val_loss', 'val_acc'],
            'showlegend': True, 
            })


print("training done !")


## Save checkpoint
net.eval()
state = {
    'state_dict': net.state_dict(),
}
save_checkpoint(state, checkpoint_path)


## Test
data = np.load(opt.test_data_path)
label = np.load(opt.test_label_path)
test_loader = load_EEG_Datasets(data, label, batch_size=opt.batch_size, is_val=False)
print("=="*20)
print("testing on {} ...".format(device))
net.eval()
loss, acc = [], []
for i, (X, y) in enumerate(test_loader):
    y_hat = net(X)
    loss_fn = criterion(y_hat, y)
    loss.append(loss_fn.item())
    acc.append(Accuracy(y_hat, y))

print("testing loss: %.4f, testing accuracy: %.4f"%(sum(loss) / len(loss), sum(acc) / len(acc)))
