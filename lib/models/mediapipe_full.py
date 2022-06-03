import logging
import numpy as np
import torch.nn as nn

from lib.models.basic import Basic
from lib.models.decoder import IterativeHeadDecoder, CoarseRefineDecoder
from lib.modules.blocks import ConvBNReLU

logger = logging.getLogger(__name__)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        if inp == oup:
            self.use_res_connect = True
        else:
            self.use_res_connect = False

        if (self.stride == 2) and (inp == oup):
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:  # stride == 1
            self.pool = None

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
                    kernel_size=kernel_size,
                    stride=self.stride,
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
            if self.stride == 2:
                return self.pool(x) + self.conv(x)
            else:  # self.stride == 1
                return x + self.conv(x)
        else:
            return self.conv(x)


class MediapipeFull(Basic):
    def __init__(self, cfg, **kwargs):
        super(MediapipeFull, self).__init__()

        block = InvertedResidual
        inverted_residual_settig_root = [
            [4, 24, 3, 2],  # 1/4
            [6, 24, 3, 1],
        ]
        inverted_residual_setting_stage2 = [
            [6, 40, 5, 2],  # 1/8
            [6, 40, 5, 1],
        ]
        inverted_residual_setting_stage3 = [
            [6, 80, 3, 2],  # 1/16
            [6, 80, 3, 1],
            [6, 80, 3, 1],
            [6, 112, 5, 1],
            [6, 112, 5, 1],
            [6, 112, 5, 1],
        ]
        inverted_residual_setting_stage4 = [
            [6, 192, 5, 2],  # 1/32
            [6, 192, 5, 1],
            [6, 192, 5, 1],
            [6, 192, 5, 1],
        ]

        # root: 1/4 resolution
        input_channel = 3
        hidden_channel = 24
        output_channel = 16
        features_root = list()
        features_root.extend(
            [
                ConvBNReLU(input_channel, hidden_channel, kernel_size=3, stride=2),
                ConvBNReLU(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=3,
                    stride=1,
                    groups=hidden_channel,
                ),
                nn.Conv2d(
                    hidden_channel, output_channel, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(output_channel),
            ]
        )

        input_channel = output_channel
        for (
            expand_ratio,
            output_channel,
            kernel_size,
            stride,
        ) in inverted_residual_settig_root:
            features_root.append(
                block(
                    input_channel,
                    output_channel,
                    kernel_size,
                    stride,
                    expand_ratio=expand_ratio,
                )
            )
            input_channel = output_channel
        self.root = nn.Sequential(*features_root)

        # stage2: 1/8 resolution
        features_stage2 = list()
        for (
            expand_ratio,
            output_channel,
            kernel_size,
            stride,
        ) in inverted_residual_setting_stage2:
            features_stage2.append(
                block(
                    input_channel,
                    output_channel,
                    kernel_size,
                    stride,
                    expand_ratio=expand_ratio,
                )
            )
            input_channel = output_channel
        self.stage2 = nn.Sequential(*features_stage2)

        # stage3: 1/16 resolution
        features_stage3 = list()
        for (
            expand_ratio,
            output_channel,
            kernel_size,
            stride,
        ) in inverted_residual_setting_stage3:
            features_stage3.append(
                block(
                    input_channel,
                    output_channel,
                    kernel_size,
                    stride,
                    expand_ratio=expand_ratio,
                )
            )
            input_channel = output_channel
        self.stage3 = nn.Sequential(*features_stage3)

        # stage4: 1/32 resolution
        features_stage4 = list()
        for (
            expand_ratio,
            output_channel,
            kernel_size,
            stride,
        ) in inverted_residual_setting_stage4:
            features_stage4.append(
                block(
                    input_channel,
                    output_channel,
                    kernel_size,
                    stride,
                    expand_ratio=expand_ratio,
                )
            )
            input_channel = output_channel

        output_channel = input_channel * 6
        features_stage4.extend(
            [
                ConvBNReLU(input_channel, output_channel, kernel_size=1, stride=1),
                ConvBNReLU(
                    output_channel,
                    output_channel,
                    kernel_size=3,
                    stride=1,
                    groups=output_channel,
                ),
            ]
        )

        self.stage4 = nn.Sequential(*features_stage4)

        # Decoder
        in_channels = [192 * 6, 112, 40, 24]
        out_channel = cfg.MODEL.EXTRA.NUM_DECONV_FILTERS

        if cfg.MODEL.EXTRA.DECODER == "IterativeHeadDecoder":
            self.final = IterativeHeadDecoder(in_channels, cfg.MODEL.NUM_JOINTS)
        elif cfg.MODEL.EXTRA.DECODER == "CoarseRefineDecoder":
            upsample_size = [
                int(cfg.MODEL.HEATMAP_SIZE[0]),
                int(np.ceil(0.5 * cfg.MODEL.HEATMAP_SIZE[0])),
                int(np.ceil(0.25 * cfg.MODEL.HEATMAP_SIZE[0])),
            ]
            self.final = CoarseRefineDecoder(
                in_channels, out_channel, cfg.MODEL.NUM_JOINTS, upsample_size
            )

    def forward(self, x):
        x1 = self.root(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x = self.final([x4, x3, x2, x1])
        return x


def get_pose_net(cfg, is_train, **kwargs):
    model = MediapipeFull(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
        logger.info(f"=> model init!")

    return model
