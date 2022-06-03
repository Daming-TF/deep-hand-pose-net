from torch import nn

from lib.modules.blocks import ConvBNReLU


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # round: 返回浮点数x的四舍五入值
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # point-wise
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
            )

        layers.extend(
            [
                # depth-wise
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                nn.Conv2d(
                    hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False
                ),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
