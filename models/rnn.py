import torch
from torch import nn
import copy

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(3 * 32 * 32, 8192, batch_first=True)
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
    model = RNN()
    clone = copy.deepcopy(model)
    inputs = torch.ones(8, 10, 3, 32, 32)
    outputs = model(inputs)
    loss = outputs.sum()
    print(loss.backward)
    cout = clone(inputs)
    loss1 = cout.sum()
    #cout2 = clone(inputs)
    #loss2 = cout2.sum()
    print(loss1.backward)
    #print(loss2.backward)


