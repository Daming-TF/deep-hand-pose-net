import torch.nn as nn
import numpy as np
import logging

from lib.models.basic import Basic
from lib.modules.blocks import ConvBNReLU
from lib.models.decoder import IterativeHeadDecoder, CoarseRefineDecoder, ClassifierDecoder
from lib.modules.mobilenetv2 import InvertedResidual


logger = logging.getLogger(__name__)


class MobileNetV2(Basic):
    def __init__(self, cfg, **kwargs):
        super(MobileNetV2, self).__init__()

        net_type = cfg.MODEL.EXTRA.NET_TYPE
        sflite_spec = {"v0": 0.35, "v1": 0.5, "v2": 0.75, "v3": 1.0, "v4": 1.4}
        self.alpha = sflite_spec[net_type]

        block = InvertedResidual
        norm_layer = nn.BatchNorm2d
        round_nearest = 8
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting_root = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],  # 1/4
        ]
        inverted_residual_setting_stage2 = [
            [6, 32, 3, 2],  # 1/8
        ]
        inverted_residual_setting_stage3 = [
            [6, 64, 4, 2],  # 1/16
            [6, 96, 3, 1],
        ]
        inverted_residual_setting_stage4 = [
            [6, 160, 3, 2],
            [6, 320, 1, 1],  # 1/32
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting_root) == 0
            or len(inverted_residual_setting_root[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting_root)
            )

        # building first layer
        input_channel = self._make_divisible(input_channel * self.alpha, round_nearest)
        self.last_channel = self._make_divisible(
            last_channel * max(1.0, self.alpha), round_nearest
        )

        # root: 1/4 resolution
        features_root = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting_root:
            output_channel = self._make_divisible(c * self.alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_root.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        self.root = nn.Sequential(*features_root)

        # stage2: 1/8 resolution
        features_stage2 = list()
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting_stage2:
            output_channel = self._make_divisible(c * self.alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_stage2.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        self.stage2 = nn.Sequential(*features_stage2)

        # stage3: 1/16 resolution
        features_stage3 = list()
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting_stage3:
            output_channel = self._make_divisible(c * self.alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_stage3.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        self.stage3 = nn.Sequential(*features_stage3)

        # stage4: 1/32 resolution
        features_stage4 = list()
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting_stage4:
            output_channel = self._make_divisible(c * self.alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features_stage4.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        # building last several layers
        features_stage4.append(
            ConvBNReLU(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer
            )
        )
        # make it nn.Sequential
        self.stage4 = nn.Sequential(*features_stage4)

        # Decoder
        in_channels = [
            self.last_channel,
            self._make_divisible(96 * self.alpha, round_nearest),
            self._make_divisible(32 * self.alpha, round_nearest),
            self._make_divisible(24 * self.alpha, round_nearest),
        ]
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
        elif cfg.MODEL.EXTRA.DECODER == "ClassifierDecoder":
            self.final = ClassifierDecoder()

    @staticmethod
    def _make_divisible(v, divisor, min_value=None):        # input_channel * self.alpha, 8
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
        if min_value is None:
            min_value = divisor

        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x1 = self.root(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x = self.final([x4, x3, x2, x1])
        return x


def get_pose_net(cfg, is_train, **kwargs):
    model = MobileNetV2(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
        logger.info(f"=> model init!")

    return model
