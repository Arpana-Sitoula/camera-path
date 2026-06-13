import torch
import torch.nn.functional as F


class SkipConnectionConcat(torch.nn.Module):
    def __init__(self):
        super(SkipConnectionConcat, self).__init__()

    def forward(self, input_list):
        return torch.cat(input_list, dim=1)


class AddConnectionConcat(torch.nn.Module):
    def __init__(self):
        super(AddConnectionConcat, self).__init__()

    def forward(self, input_list):
        return torch.add(*input_list)


class Multiplication(torch.nn.Module):
    def __init__(self):
        super(Multiplication, self).__init__()

    def forward(self, input_list):
        return torch.mul(*input_list)


class View(torch.nn.Module):
    def __init__(self):
        super(View, self).__init__()
        self.shape = []

    def forward(self, input, shape):
        self.shape = shape
        return input.view(shape)


class Interpolate(torch.nn.Module):
    def __init__(self):
        super(Interpolate, self).__init__()
        self.shape = []

    def forward(self, input, shape):
        self.shape = shape
        return F.interpolate(input, shape, mode='bilinear', align_corners=False)


class Padding(torch.nn.Module):

    def __init__(self, padding):
        super(Padding, self).__init__()
        self.p = padding

    def forward(self, x):
        x = F.pad(x, (self.p, self.p, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.p, self.p), mode='replicate')
        return x