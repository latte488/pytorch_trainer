import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(3 * 32 * 32, 1024, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 8),
        )   
    def forward(self, x): 
        x = torch.flatten(x, 2)
        x, _  = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x

if __name__ == '__main__':
    model = RNN()
    inputs = torch.randn(8, 10, 3, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
