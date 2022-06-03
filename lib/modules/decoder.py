import torch
import torch.nn as nn

from lib.modules import InvertedResidual


class CoarseModule(nn.Module):
    def __init__(self, channel_settings, out_channel, upsample_size):
        super(CoarseModule, self).__init__()

        self.channel_settings = channel_settings
        laterals, upsamples = [], []
        for i in range(len(self.channel_settings)):
            laterals.append(self._lateral(self.channel_settings[i], out_channel))

            if i != len(self.channel_settings) - 1:
                upsamples.append(self._upsample(out_channel, upsample_size[i]))

        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)

    @staticmethod
    def _lateral(in_channel, out_channel):
        layers = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        return layers

    @staticmethod
    def _upsample(out_channel, upsample_size):
        layers = nn.Sequential(
            nn.Upsample(size=(upsample_size, upsample_size), mode="nearest"),
            InvertedResidual(out_channel, out_channel, stride=1),
        )
        return layers

    def forward(self, x):
        coarse_fms = []
        up = None

        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            coarse_fms.append(feature)

            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)

        return coarse_fms


class RefineModule(nn.Module):
    def __init__(self, in_channel, out_channel, num_cascade, upsample_size):
        super(RefineModule, self).__init__()

        cascade = []
        self.num_cascade = num_cascade
        for i in range(self.num_cascade - 1):
            cascade.append(
                self._make_layer(
                    in_channel, self.num_cascade - i - 1, out_channel, upsample_size[i:]
                )
            )

        self.cascade = nn.ModuleList(cascade)
        self.final = nn.Sequential(
            nn.Conv2d(
                out_channel * self.num_cascade,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_layer(in_channel, num, out_channel, upsample_size):
        layers = []
        for i in range(num):
            layers.append(InvertedResidual(in_channel, out_channel, stride=1))
            layers.append(InvertedResidual(out_channel, out_channel, stride=1))
            layers.append(
                nn.Upsample(size=(upsample_size[i], upsample_size[i]), mode="nearest")
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(self.num_cascade):
            if i == self.num_cascade - 1:
                refine_fms.append(x[i])
            else:
                refine_fms.append(self.cascade[i](x[i]))

        out = torch.cat(refine_fms, dim=1)
        out = self.final(out)

        return out
