import logging
import numpy as np
import torch.nn as nn

from lib.modules.activations import Hswish
from lib.models.basic import Basic
from lib.models.decoder import IterativeHeadDecoder, CoarseRefineDecoder
from lib.modules.mobilenetv3 import Block, SeModule


logger = logging.getLogger(__name__)


class MobileNetV3Large(Basic):
    def __init__(self, cfg, **kwargs):
        super(MobileNetV3Large, self).__init__()

        net_type = cfg.MODEL.EXTRA.NET_TYPE
        sflite_spec = {
            "v0": 0.35,
            "v1": 0.5,
            "v2": 0.75,
            "v3": 1.0,
            "v4": 1.25,
            "v5": 1.40,
        }
        self.alpha = sflite_spec[net_type]

        self.root = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=int(16 * self.alpha),
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(16 * self.alpha)),
            Hswish(),
            Block(
                kernel_size=3,
                in_size=int(16 * self.alpha),
                expand_size=int(16 * self.alpha),
                out_size=int(16 * self.alpha),
                nolinear=nn.ReLU(inplace=True),
                semodule=None,
                stride=1,
            ),
            Block(
                kernel_size=3,
                in_size=int(16 * self.alpha),
                expand_size=int(64 * self.alpha),
                out_size=int(24 * self.alpha),
                nolinear=nn.ReLU(inplace=True),
                semodule=None,
                stride=2,
            ),
            Block(
                kernel_size=3,
                in_size=int(24 * self.alpha),
                expand_size=int(72 * self.alpha),
                out_size=int(24 * self.alpha),
                nolinear=nn.ReLU(inplace=True),
                semodule=None,
                stride=1,
            ),
        )  # 1/4

        self.stage2 = nn.Sequential(
            Block(
                kernel_size=5,
                in_size=int(24 * self.alpha),
                expand_size=int(72 * self.alpha),
                out_size=int(40 * self.alpha),
                nolinear=nn.ReLU(inplace=True),
                semodule=SeModule(in_size=int(40 * self.alpha)),
                stride=2,
            ),
            Block(
                kernel_size=5,
                in_size=int(40 * self.alpha),
                expand_size=int(120 * self.alpha),
                out_size=int(40 * self.alpha),
                nolinear=nn.ReLU(inplace=True),
                semodule=SeModule(in_size=int(40 * self.alpha)),
                stride=1,
            ),
            Block(
                kernel_size=5,
                in_size=int(40 * self.alpha),
                expand_size=int(120 * self.alpha),
                out_size=int(40 * self.alpha),
                nolinear=nn.ReLU(inplace=True),
                semodule=SeModule(in_size=int(40 * self.alpha)),
                stride=1,
            ),
        )  # 1/8

        self.stage3 = nn.Sequential(
            Block(
                kernel_size=3,
                in_size=int(40 * self.alpha),
                expand_size=int(240 * self.alpha),
                out_size=int(80 * self.alpha),
                nolinear=Hswish(),
                semodule=None,
                stride=2,
            ),
            Block(
                kernel_size=3,
                in_size=int(80 * self.alpha),
                expand_size=int(200 * self.alpha),
                out_size=int(80 * self.alpha),
                nolinear=Hswish(),
                semodule=None,
                stride=1,
            ),
            Block(
                kernel_size=3,
                in_size=int(80 * self.alpha),
                expand_size=int(184 * self.alpha),
                out_size=int(80 * self.alpha),
                nolinear=Hswish(),
                semodule=None,
                stride=1,
            ),
            Block(
                kernel_size=3,
                in_size=int(80 * self.alpha),
                expand_size=int(184 * self.alpha),
                out_size=int(80 * self.alpha),
                nolinear=Hswish(),
                semodule=None,
                stride=1,
            ),
            Block(
                kernel_size=3,
                in_size=int(80 * self.alpha),
                expand_size=int(480 * self.alpha),
                out_size=int(112 * self.alpha),
                nolinear=Hswish(),
                semodule=SeModule(in_size=int(112 * self.alpha)),
                stride=1,
            ),
            Block(
                kernel_size=3,
                in_size=int(112 * self.alpha),
                expand_size=int(672 * self.alpha),
                out_size=int(112 * self.alpha),
                nolinear=Hswish(),
                semodule=SeModule(in_size=int(112 * self.alpha)),
                stride=1,
            ),
        )

        self.stage4 = nn.Sequential(
            Block(
                kernel_size=5,
                in_size=int(112 * self.alpha),
                expand_size=int(672 * self.alpha),
                out_size=int(160 * self.alpha),
                nolinear=Hswish(),
                semodule=SeModule(in_size=int(160 * self.alpha)),
                stride=2,
            ),
            Block(
                kernel_size=5,
                in_size=int(160 * self.alpha),
                expand_size=int(960 * self.alpha),
                out_size=int(160 * self.alpha),
                nolinear=Hswish(),
                semodule=SeModule(in_size=int(160 * self.alpha)),
                stride=1,
            ),
            Block(
                kernel_size=5,
                in_size=int(160 * self.alpha),
                expand_size=int(960 * self.alpha),
                out_size=int(160 * self.alpha),
                nolinear=Hswish(),
                semodule=SeModule(in_size=int(160 * self.alpha)),
                stride=1,
            ),
            nn.Conv2d(
                in_channels=int(160 * self.alpha),
                out_channels=int(960 * self.alpha),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(int(960 * self.alpha)),
            Hswish(),
        )

        # Decoder
        in_channels = [
            int(960 * self.alpha),
            int(112 * self.alpha),
            int(40 * self.alpha),
            int(24 * self.alpha),
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
    model = MobileNetV3Large(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
        logger.info(f"=> model init!")

    return model
