import torch
import torch.nn as nn
from torch import Tensor


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True),
        )


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    # refer to: https://github.com/pytorch/vision/blob/8265469b2526e8d520e2e5ede71454ac01e357aa/torchvision/models/shufflenetv2.py#L21
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def conv_block(inp, out, alpha, kernel_size=3, stride=1, padding=1):
    out = int(out * alpha)
    return nn.Sequential(
        nn.Conv2d(
            inp,
            out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True),
    )


def depthwise_separable_conv_block(inp, oup, alpha=1, stride=1):
    inp = int(inp * alpha)
    oup = int(oup * alpha)
    return nn.Sequential(
        nn.Conv2d(
            inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False
        ),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )
