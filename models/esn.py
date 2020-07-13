import torch
from torch import nn

if __name__ == '__main__':
    import rc
else:
    from . import rc

class ESN(nn.Module):
    def __init__(self):
        super(ESN, self).__init__()
        self.rnn = rc.ESN(3 * 32 * 32, 8192, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8192, 8),
        )

    def forward(self, x):
        x = x.flatten(2)
        x = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x

if __name__ == '__main__':
    model = ESN()
    inputs = torch.randn(8, 10, 3, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
