import torch
from torch import nn

class DNN(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.linears = nn.Sequential(
            nn.Linear(input_size, 1024),
            self.relu,
            nn.Linear(1024, 1024),
            self.relu,
            nn.Linear(1024, 1024),
            self.relu,
            nn.Linear(1024, 1024),
            self.relu,
            nn.Linear(1024, 1024),
            self.relu,
            nn.Linear(1024, 1024),
            self.relu,
            nn.Linear(1024, 1024),
            self.relu,
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.linears(x)
        return x

