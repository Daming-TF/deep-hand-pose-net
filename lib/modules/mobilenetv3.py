import torch.nn as nn
from lib.modules.activations import Hsigmoid


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_size,
                in_size // reduction,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_size // reduction,
                in_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(in_size),
            Hsigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    # expand + depthwise + pointwise
    def __init__(
        self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride
    ):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(
            in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(
            expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.se is not None:
            out = self.se(out)

        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
