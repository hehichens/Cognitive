import os
import importlib
import numpy as np 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys; sys.path.append("..")
from utils.options import opt


from models.basemodel import basemodel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


## load checkpoint
def load_checkpoint(net, checkpoint):
    print("=> load checkpoint")
    net.load_state_dict(checkpoint['state_dict'])


## save checkpoint
def save_checkpoint(state, save_path=opt.checkpoint_path):
    print("=> save checkpoint")
    torch.save(state, save_path)


## regulization parameters 
def regulization(net, lambda_=opt.lambda_):
    w = torch.cat([x.view(-1) for x in net.parameters()])
    err = lambda_ * torch.sum(torch.abs(w))
    return err


## make dirs
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


## make dir
def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


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

    
## Accuracy
def Accuracy(y_hat, y):
    pred = y_hat.max(1)[1]
    return (pred == y).sum().item() / len(pred)

## load EEG Datasets
def load_EEG_Datasets(data, label, batch_size=1, is_val=True):
    """
    data: np.array: 32 x 40 x 40 x num_dim
    label: np.array: 32 x 40
    """
    # combine 0 and 1 dimension
    data = np.concatenate(data, axis=0) # 1280X40Xnum_dim
    label = np.concatenate(label, axis=0) # 1280

    # load test data
    if is_val == False:
        dataset = EEGDataset(data, label)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        return loader

    # data size: train:val = 4:1
    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=opt.val_size, random_state=opt.seed)

    dataset_train = EEGDataset(X_train, y_train)
    dataset_val = EEGDataset(X_val, y_val)

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True, pin_memory=False)

    return train_loader, val_loader


## Dynamic Import Models
def find_model_using_name(model_name):
    """Import the module "models/[model_name].py".
    model options:
    - basemodel
    - EEGNet
    - TSception

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') #  + 'model'
    for name, cls in modellib.__dict__.items():
        # model = cls
        # break
        if name.lower() == target_model_name.lower():
            # and issubclass(cls, BaseModel)
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance



if __name__ == "__main__":
    data = np.load('../datasets/data/data.npy') # (32, 40, 40, 101)
    arousal_labels = np.load('../datasets/data/arousal_labels.npy')
    valence_labels = np.load('../datasets/data/valence_labels.npy')

    # test data
    train_loader, val_loader = load_EEG_Datasets(data, arousal_labels)
    for i, (x, y) in enumerate(train_loader):
        print(x.shape, y.shape)
        break