import torch
from torch import nn

class ConvRNNCellBase(nn.Module):
    def __init__(self, gate_size, in_size, out_size, kernel_size, bias=False):
        super(ConvRNNCellBase, self).__init__()

        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd number.')

        self.out_size = out_size
        self.conv = nn.Conv2d(
            in_channels=in_size + out_size,
            out_channels=gate_size * out_size,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias,
        )

    def initial_hx(self, x):
        b, c, h, w = x.shape
        zeros = torch.zeros(b, self.out_size, h, w,
                        dtype=x.dtype, device=x.device)
        return zeros, zeros

class ConvTanhRNNCell(ConvRNNCellBase):
    def __init__(self, *args, **kwargs):
        super(ConvTanhRNNCell, self).__init__(1, *args, **kwargs)

    def forward(self, x, hx):
        prev_h, prev_c = hx
        x = torch.cat([x, prev_h], dim=1)
        x = self.conv(x)
        x = torch.tanh(x)
        return x, x

class ConvReLURNNCell(ConvRNNCellBase):
    def __init__(self, *args, **kwargs):
        super(ConvReLURNNCell, self).__init__(1, *args, **kwargs)

    def forward(self, x, hx):
        prev_h, prev_c = hx
        x = torch.cat([x, prev_h], dim=1)
        x = self.conv(x)
        x = torch.relu(x)
        return x, x

class ConvGRUCell(ConvRNNCellBase):
    def __init__(self, *args, **kwargs):
        super(ConvGRUCell, self).__init__(3, *args, **kwargs)

    def forward(self, x, hx):
        prev_h, prev_c = hx
        x = torch.cat([x, prev_h], dim=1)
        x = self.conv(x)
        f, c, o = torch.split(x, self.out_size, dim=1)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)
        hx = f * prev_c + (1 - f) * c
        x = o * torch.tanh(c)
        return x, hx

class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self, *args, **kwargs):
        super(ConvLSTMCell, self).__init__(4, *args, **kwargs)

    def forward(self, x, hx):
        prev_h, prev_c = hx
        x = torch.cat([x, prev_h], dim=1)
        x = self.conv(x)
        f, i, c, o = torch.split(x, self.out_size, dim=1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        c = torch.tanh(c)
        o = torch.sigmoid(o)
        hx = f * prev_c + i * c
        x = o * torch.tanh(c)
        return x, hx
    
class ConvRNNBase(nn.Module):

    def __init__(self, cell_class, input_size, hidden_sizes, kernel_sizes, batch_first=False):
        super(ConvRNNBase, self).__init__()
        self.batch_first = batch_first
        self.num_layers = len(hidden_sizes)
        self.cells = [cell_class(input_size, hidden_sizes[0], kernel_sizes[0])]
        for i in range(self.num_layers - 1):
            self.cells.append(cell_class(hidden_sizes[i], hidden_sizes[i + 1], kernel_sizes[i + 1]))
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

class ConvTanhRNN(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        super(ConvTanhRNN, self).__init__(ConvTanhRNNCell, *args, **kwargs)

class ConvReLURNN(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        super(ConvReLURNN, self).__init__(ConvReLURNNCell, *args, **kwargs)

class ConvGRU(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        super(ConvGRU, self).__init__(ConvGRUCell, *args, **kwargs)

class ConvLSTM(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        super(ConvLSTM, self).__init__(ConvLSTMCell, *args, **kwargs)
