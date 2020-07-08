import torch
from torch import nn
from .esn import ESN


class CESN(nn.Module):
    def __init__(self, num_classes):
        super(CESN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 1024, 3, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.rnn = ESN(1024 * 3 * 3, 1024, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        x = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x
