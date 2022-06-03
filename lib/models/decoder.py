import torch.nn as nn
import torch.nn.functional as F


from lib.modules import InvertedResidual, CoarseModule, RefineModule
from lib.modules.blocks import depthwise_separable_conv_block


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, num_joints):
        super(SimpleDecoder, self).__init__()
        self.layers = nn.Conv2d(
            in_channels[-1], num_joints, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = x[-1]
        out = self.layers(x)
        return out


class NaiveDecoder(nn.Module):
    def __init__(self, in_channels, out_channel, num_joints):
        super(NaiveDecoder, self).__init__()
        print(f"in_channels: {in_channels}")

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels[-1],
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            InvertedResidual(out_channel, out_channel, stride=1),
            InvertedResidual(out_channel, out_channel, stride=1),
            nn.Conv2d(
                out_channel, num_joints, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x):
        x = x[-1]
        out = self.layers(x)
        return out


class IterativeHeadDecoder(nn.Module):
    def __init__(self, in_channels, num_joints):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels

        for i in range(num_branchs):
            if i != num_branchs - 1:
                out_channel = self.in_channels[i + 1]
            else:
                out_channel = self.in_channels[i]
            projects.append(
                depthwise_separable_conv_block(
                    self.in_channels[i], out_channel, stride=1
                )
            )
        self.projects = nn.ModuleList(projects)

        self.final = nn.Conv2d(
            in_channels=self.in_channels[-1],
            out_channels=num_joints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):       # [x4, x3, x2, x1]
        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x, size=s.size()[-2:], mode="bilinear", align_corners=True
                )
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        out = self.final(y[-1])
        return out


class CoarseRefineDecoder(nn.Module):
    def __init__(self, in_channels, out_channel, num_joints, upsample_size):
        super().__init__()

        self.coarse = CoarseModule(in_channels, out_channel, upsample_size[::-1])
        self.refine = RefineModule(
            out_channel, out_channel, len(in_channels), upsample_size[::-1]
        )

        self.final = nn.Sequential(
            InvertedResidual(out_channel, out_channel, stride=1),
            InvertedResidual(out_channel, out_channel, stride=1),
            nn.Conv2d(
                out_channel, num_joints, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x):
        x_list = self.coarse(x)
        x = self.refine(x_list)
        out = self.final(x)
        return out


class ClassifierDecoder(nn.Module):
    def __init__(self, out_channel, num_classes, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            # building classifier
            nn.Dropout(p=dropout),
            nn.Linear(out_channel, num_classes),
        )

    def forward(self, x):
        return self.layers(x)