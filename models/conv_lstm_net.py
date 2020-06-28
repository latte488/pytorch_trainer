import torch
from torch import nn
from .convrnn import ConvLSTM

class ConvLSTMNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvLSTMNet, self).__init__()
        self.convrnn = ConvLSTM(3, [256], [3], batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3)) 
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(256 * 3 * 3, num_classes)

    def forward(self, x):
        x, _ = self.convrnn(x)
        x = self.avgpool(x[:, -1, :])
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
