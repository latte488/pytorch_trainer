import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(3 * 32 * 32, 8192, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 8),
        )   

    def forward(self, x): 
        x = torch.flatten(x, 2)
        x, _  = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x

if __name__ == '__main__':
    model = LSTM()
    inputs = torch.randn(8, 10, 3, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
