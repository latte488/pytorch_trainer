import torch
from torch import nn
from .esn import ESN

class ESNNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ESNNet, self).__init__()
        self.rnn = ESN(input_size, 1024, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        x = x.flatten(2)
        x = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x

