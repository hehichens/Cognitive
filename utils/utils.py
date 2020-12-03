import numpy as np 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys; sys.path.append("..")
from utils.options import opt


from models.basemodel import CNN_DEAP
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


## load checkpoint
def load_checkpoint(checkpoint):
    print("=> load checkpoint")
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


## save checkpoint
def save_checkpoint(state, save_path=opt.checkpoint_path):
    print("=> save checkpoint")
    torch.save(state, save_path)


## EEGDataset Object
class EEGDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x[:,np.newaxis,:,:]
        self.y = y
        assert self.x.shape[0] == self.y.shape[0]

    def __getitem__(self, index):
        y = torch.tensor(self.y[index]).long().to(device)
        x = torch.tensor(self.x[index]).float().to(device)
        return x, y

    def __len__(self):
        return len(self.y)


## load EEG Datasets
def load_EEG_Datasets(data, label, batch_size=1):
    """
    data: np.array ==> 32X40X40X101
    label: np.array ==> 32X40
    """
    # combine 0 and 1 dimension
    data = np.concatenate(data, axis=0) # 1280X40X101
    label = np.concatenate(label, axis=0) # 1280

    # data size: train > val > test
    X_train, X_test, y_train, y_test = train_test_split(data, label)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test)

    dataset_train = EEGDataset(X_train, y_train)
    dataset_val = EEGDataset(X_val, y_val)
    dataset_test = EEGDataset(X_test, y_test)

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size,shuffle=True, pin_memory=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, pin_memory=False)

    return train_loader, val_loader, test_loader


## train network
def train(net, data, label, num_epochs, criterion, optimizer):
    """
        net: networks
        data: np.array
        label: np.array
        num_epochs: number of epochs
        criterion: loss function
        optimzier: optimzier

        egg:
        >>> net = CNN_DEAP().to(device)
        >>> num_epochs = 10
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = optim.Adam(net.parameters()) # lr defualt:1e-3
        >>> train(net, data, arousal_labels, num_epochs, criterion, optimizer)
    """
    train_loader, val_loader, test_loader = load_EEG_Datasets(data, label)

    print("training on {} ...".format(device))
    for epoch in range(num_epochs):
        train_loss, train_acc = [], []
        for i, (X, y) in tqdm(enumerate(train_loader)):
            y_hat = net(X) # batch_size X 2
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = y_hat.max(1)[1]
            acc = (pred == y).sum().item() / len(pred)

            train_loss.append(loss)
            train_acc.append(acc)

        print("epoch: %d, training loss: %.4f, training accuracy: %.4f"%(
            epoch+1, sum(train_loss) / len(train_loss), sum(train_acc) / len(train_acc)
        ))





if __name__ == "__main__":
    data = np.load('../datasets/data/data.npy') # (32, 40, 40, 101)
    arousal_labels = np.load('../datasets/data/arousal_labels.npy')
    valence_labels = np.load('../datasets/data/valence_labels.npy')

    # test data
    # train_loader, val_loader, test_loader = load_EEG_Datasets(data, arousal_labels)
    # for i, (x, y) in enumerate(train_loader):
    #     print(x.shape, y.shape)
    #     break

    # test model
    net = CNN_DEAP().to(device)
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters()) # lr defualt:1e-3
    train(net, data, arousal_labels, num_epochs, criterion, optimizer)