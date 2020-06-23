import torch
from torch import nn

class GRUNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(GRUNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.rnn = nn.GRU(
            input_size=input_size, 
            hidden_size=1024,
            dropout=0.5,
            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1024),
            self.relu,
            nn.Dropout(),
            nn.Linear(1024, 1024),
            self.relu,
            nn.Dropout(),
            nn.Linear(1024, 1024),
            self.relu,
            nn.Dropout(),
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        x = x.flatten(2)
        x, _ = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x
