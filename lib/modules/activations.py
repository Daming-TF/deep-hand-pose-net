import torch.nn as nn
import torch.nn.functional as F


class Hswish(nn.Module):
    def __init__(self):
        super(Hswish, self).__init__()
        pass

    @staticmethod
    def forward(x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class Hsigmoid(nn.Module):
    def __init__(self):
        super(Hsigmoid, self).__init__()
        pass

    @staticmethod
    def forward(x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
