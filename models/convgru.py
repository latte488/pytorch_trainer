import torch
from torch import nn
from torchvision import models

class ResBlock(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.conv3 = nn.Conv2d(hidden_channels, input_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(input_channels)

        self.conv4 = nn.Conv2d(input_channels, hidden_channels, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(hidden_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
    
        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        return x
    
class ResGRUCell(nn.Module):

    def __init__(self, input_size, hidden_sizes, out_size):
        super(ResGRUCell, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        in_channels = input_size + out_size
        num_layers = len(hidden_sizes)
        out_channels = 3 * out_size

        self.in_conv = nn.Conv2d(in_channels, hidden_sizes[0], 7, stride=1, padding=3, bias=False)
        self.in_bn = nn.BatchNorm2d(hidden_sizes[0])

        layers = [ResBlock(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(num_layers - 1)]
        self.hidden_layers = nn.ModuleList(layers)

        self.out_conv = nn.Conv2d(hidden_sizes[-1], out_channels, 3, padding=1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, hx):
        prev_h, prev_c = hx
        identity = torch.cat([prev_h, prev_h, prev_h], dim=1)

        x = torch.cat([x, prev_h], dim=1)
        x = self.in_conv(x)
        x = self.in_bn(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        x = self.out_conv(x)
        x = self.out_bn(x)

        x += identity
        f, c, o = torch.split(x, self.out_size, dim=1)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)

        c = f * prev_c + (1 - f) * c
        h = o * torch.tanh(c)
        return h, c

    def initial_hx(self, x):
        b, c, h, w = x.shape
        zeros = torch.zeros(b, self.out_size, h, w,
                        dtype=x.dtype, device=x.device)
        return zeros, zeros

class ResRGU(nn.Module):

    def __init__(self, input_size, hidden_sizes, out_size, batch_first=False):
        super(ResRGU, self).__init__()
        self.batch_first = batch_first
        self.rnn = ResGRUCell(input_size, hidden_sizes, out_size)

    def forward(self, xs):
        if self.batch_first:
            xs = xs.permute(1, 0, 2, 3, 4)
        hx = self.rnn.initial_hx(xs[0])
        for x in xs:
            hx = self.rnn(x, hx)
        h, c = hx
        return h, (h, c)
                
class ResRGUNet(nn.Module):
    
    def __init__(self, num_classes):
        super(ResRGUNet, self).__init__()
        self.convrnn = ResRGU(3, [64, 64, 256, 256, 512], 512, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3)) 
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x, _ = self.convrnn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
