import torch
import torch.optim as optim
import torch.nn as nn

class CNN_DEAP(nn.Module):
    def __init__(self, num_class, input_size):
        super(CNN_DEAP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 100 ,kernel_size=3),
            nn.Tanh(),
            nn.Conv2d(100, 100, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, ),
            nn.Dropout(p=0.5)
        )

        self.size = self.getsize(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.size[1], 128),
            nn.Tanh(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_class)
            )

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # print("size:"+str(self.size))
        # print(x.shape)
        x = self.classifier(x)
        return x

    def getsize(self, input_size):
        data = torch.ones(1, 1, input_size[0], input_size[1])
        x = self.features(data)
        out = x.view(x.shape[0], -1)
        return out.size()


if __name__ == "__main__":
    model = CNN_DEAP(2, [40, 101])
    print(model)