import torch
from torch import nn

class LstmNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LstmNet, self).__init__()
        self.features = nn.LSTM(
            input_size=input_size,
            hidden_size=10,
            num_layers=3,
            batch_first=True,
        )   
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_classes),
        )   

    def forward(self, x): 
        x = torch.flatten(x, 2)
        x, _  = self.features(x)
        x = self.classifier(x[:, -1, :]) 
        return x

class LstmNet2(nn.Module):

    def __init__(self, input_size, num_classes):
        super(LstmNet2, self).__init__()
        self.features = nn.LSTM(
            input_size=input_size,
            hidden_size=10,
            num_layers=3,
            batch_first=True,
        )   
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_classes),
        )   

    def forward(self, x): 
        b, t = x.shape
        x = x.view(b, t, 1)
        x, _  = self.features(x)
        x = self.classifier(x[:, -1, :]) 
        return x
