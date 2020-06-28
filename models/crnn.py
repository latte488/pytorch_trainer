import torch
from torch import nn
from torchvision import models

torch.autograd.set_detect_anomaly(True)

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 1024, 3, bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.rnn = nn.LSTM(1024 * 3 * 3, 2048, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        x, _ = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x

