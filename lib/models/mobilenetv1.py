import logging
import torch.nn as nn
import numpy as np

from lib.models.basic import Basic
from lib.models.decoder import IterativeHeadDecoder, CoarseRefineDecoder
from lib.modules.blocks import conv_block, depthwise_separable_conv_block


logger = logging.getLogger(__name__)


class MobileNet(Basic):
    def __init__(self, cfg, **kwargs):
        super(MobileNet, self).__init__()

        net_type = cfg.MODEL.EXTRA.NET_TYPE
        sflite_spec = {
            "v0": 0.25,
            "v1": 0.5,
            "v2": 0.75,
            "v3": 1.0,
            "v4": 1.25,
            "v5": 1.40,
        }
        self.alpha = sflite_spec[net_type]

        self.root = nn.Sequential(
            conv_block(3, 32, alpha=self.alpha, stride=2),
            depthwise_separable_conv_block(32, 64, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(64, 128, alpha=self.alpha, stride=2),  # 1/4
        )

        self.stage2 = nn.Sequential(
            depthwise_separable_conv_block(128, 128, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(128, 256, alpha=self.alpha, stride=2),  # 1/8
        )

        self.stage3 = nn.Sequential(
            depthwise_separable_conv_block(256, 256, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(
                256, 512, alpha=self.alpha, stride=2
            ),  # 1/16
        )

        self.stage4 = nn.Sequential(
            depthwise_separable_conv_block(512, 512, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(512, 512, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(512, 512, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(512, 512, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(512, 512, alpha=self.alpha, stride=1),
            depthwise_separable_conv_block(
                512, 1024, alpha=self.alpha, stride=2
            ),  # 1/32
            depthwise_separable_conv_block(1024, 1024, alpha=self.alpha, stride=1),
        )

        # Decoder
        in_channels = [
            int(1024 * self.alpha),
            int(512 * self.alpha),
            int(256 * self.alpha),
            int(128 * self.alpha),
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

    def forward(self, x):
        x1 = self.root(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x = self.final([x4, x3, x2, x1])
        return x


def get_pose_net(cfg, is_train, **kwargs):
    model = MobileNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
        logger.info(f"=> model init!")

    return model
