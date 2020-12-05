import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import sys; sys.path.append("..")

class basemodel(nn.Module):
    def __init__(self, opt):
        num_class = opt.num_class
        input_size = [opt.num_channel, opt.num_dim]
        super(basemodel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10 ,kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5)
          
        )
        self.size = self.getsize(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.size[1], 128),
            nn.Tanh(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_class)
        )
           
    def forward(self, x):
        x = self.features(x)
        # print("x = self.features(x)", x.shape)
        # sys.exit(0)
        # print("size:"+str(self.size))
        x = self.classifier(x)
        return x
    
    def getsize(self, input_size):
        data = torch.ones(1, 1, input_size[0], input_size[1])
        x = self.features(data) 
        out = x.view(x.shape[0], -1)
        return out.size()


if __name__ == "__main__":
    from utils.options import opt
    model = basemodel(opt)
    # print(model)
    x = torch.rand(2, 1, 40, 101) # batchsize, trial, channels, data
    print(model(x))