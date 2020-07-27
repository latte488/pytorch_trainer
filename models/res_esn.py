import torch
from torch import nn

if __name__ == '__main__':
    import rc
    import resnet
else:
    from . import rc
    from . import resnet

class ResESN(nn.Module):
    def __init__(self):
        super(ResESN, self).__init__()
        self.cnn = resnet.resnet50(pretrained=True)
        self.aap = nn.AdaptiveAvgPool2d((2, 3))
        self.rnn = rc.ESN(2048 * 2 * 3, 1024, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 174),
        )

    def forward(self, x):
        if self.training:
            self.eval()
            with torch.no_grad():
                b, t, c, h, w = x.shape
                x = x.view(-1, c, h, w)
                x = self.cnn(x)
                x = self.aap(x)
                x = x.detach()
            self.train()
        else:
            b, t, c, h, w = x.shape
            x = x.view(-1, c, h, w)
            x = self.cnn(x)
            x = self.aap(x)
            x = x.detach()

        x = x.view(b, t, -1)
        x = self.rnn(x)
        x = self.classifier(x[:, -1, :])
        return x

if __name__ == '__main__':
    model = CESN().to('cuda')
    inputs = torch.randn(8, 10, 3, 240//6, 497//6).to('cuda')
    outputs = model(inputs)
    print(outputs.shape)
