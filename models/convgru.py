import torch
from torch import nn
from torchvision import models

class ConvGRUCell(nn.Module):

    def __init__(self, input_size, out_size, kernel_size, bias=False):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        in_channels = input_size + out_size
        out_channels = 3 * out_size

        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd number.')

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, hx):
        prev_h, prev_c = hx

        x = torch.cat([x, prev_h], dim=1)
        x = self.conv(x)
        x = self.bn(x)

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

class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, batch_first=False):
        super(ConvGRU, self).__init__()
        self.batch_first = batch_first
        self.num_layers = len(hidden_sizes)
        self.cells = [ConvGRUCell(input_size, hidden_sizes[0], kernel_sizes[0])]
        for i in range(self.num_layers - 1):
            self.cells.append(ConvGRUCell(hidden_sizes[i], hidden_sizes[i + 1], kernel_sizes[i + 1]))
        self.cells = nn.ModuleList(self.cells)

    def forward(self, xs, hxs=None):
        if self.batch_first:
            xs = xs.permute(1, 0, 2, 3, 4)

        if hxs == None:
            xs, hxs = self.first_forward(xs)
        else:
            xs, hxs = self.continue_forward(hxs)

        if self.batch_first:
            xs = xs.permute(1, 0, 2, 3, 4)
        return xs, hxs


    def first_forward(self, xs):
        hxs = []
        for cell in self.cells:
            hx = cell.initial_hx(xs[0])
            outs = []
            for x in xs:
                hx = cell(x, hx)
                h, c = hx 
                outs.append(h)
            hxs.append(hx)
            xs = outs
        xs = torch.stack(xs)
        return xs, hxs

    def continue_forward(self, xs, hxs):
        for i in range(self.num_layers):
            cell = self.cells[i]
            hx = hxs[i]
            outs = []
            for x in xs:
                hx = cell(x, hx)
                h, c = hx 
                outs.append(h)
            hxs[i] = hx
            xs = outs
        xs = torch.stack(xs)
        return xs, hxs
                
class ConvGRUNet(nn.Module):
    
    def __init__(self, num_classes):
        super(ConvGRUNet, self).__init__()
        self.convrnn = ConvGRU(3, [256], [3], batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3)) 
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x, _ = self.convrnn(x)
        x = self.avgpool(x[:, -1, :])
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
