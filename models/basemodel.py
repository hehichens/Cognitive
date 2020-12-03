import torch
import torch.optim as optim
import torch.nn as nn

import sys

class CNN_DEAP(nn.Module):
    def __init__(self, num_class=2, input_size=[40, 101]):
        super(CNN_DEAP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 100 ,kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(100, 100, kernel_size=3, padding=1),
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
        # print("self.size", self.size)
        # sys.exit(0)
        # print("size:"+str(self.size))
        x = self.classifier(x)
        return x
    
    def getsize(self, input_size):
        data = torch.ones(1, 1, input_size[0], input_size[1])
        x = self.features(data) # bug
        out = x.view(x.shape[0], -1)
        return out.size()


if __name__ == "__main__":
    model = CNN_DEAP(2, [40, 101])
    print(model)
    print("=="*20)
    x = torch.rand(1, 1, 40, 101) # batchsize, trial, channels, data
    print(x.shape)
    print(model(x))