import logging
import numpy as np
import torch.nn as nn

from lib.modules import InvertedResidual, CoarseModule, RefineModule
from lib.models.basic import Basic


logger = logging.getLogger(__name__)


class SFLite(Basic):
    def __init__(self, cfg, **kwargs):
        super(SFLite, self).__init__()
        net_type = cfg.MODEL.EXTRA.NET_TYPE

        sflite_spec = {
            "v0": ([4, 8, 4], [32, 64, 128, 256, 256]),
            "v1": ([4, 8, 4], [32, 64, 128, 192, 256]),
            "v2": ([4, 4, 4], [32, 64, 128, 192, 256]),
            "v3": ([4, 8, 4], [32, 64, 128, 256, 384]),
            "v4": ([4, 8, 4], [24, 48, 96, 192, 256]),
            "v5": ([4, 8, 4], [16, 32, 64, 128, 256]),
        }
        stages_repeats, channel_settings = sflite_spec[net_type]

        # Encoder: Root -> Stem -> Stage2 -> Stage3 -> Stage4
        # Root
        in_channel = 3
        out_channel = channel_settings[0]
        self.root = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        # Stem
        in_channel = out_channel
        out_channel = channel_settings[1]
        self.stem = nn.Sequential(
            InvertedResidual(in_channel, out_channel, stride=2),
            InvertedResidual(out_channel, out_channel, stride=1),
            InvertedResidual(out_channel, out_channel, stride=1),
            InvertedResidual(out_channel, out_channel, stride=1),
        )

        # stage2~4
        in_channel = out_channel
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, out_channel in zip(
            stage_names, stages_repeats, channel_settings[2:]
        ):
            seq = [InvertedResidual(in_channel, out_channel, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(out_channel, out_channel, 1))
            setattr(self, name, nn.Sequential(*seq))
            in_channel = out_channel

        # Decoder
        in_channels = channel_settings[1:][::-1]
        out_channel = cfg.MODEL.EXTRA.NUM_DECONV_FILTERS
        upsample_size = [
            int(cfg.MODEL.HEATMAP_SIZE[0]),
            int(np.ceil(0.5 * cfg.MODEL.HEATMAP_SIZE[0])),
            int(np.ceil(0.25 * cfg.MODEL.HEATMAP_SIZE[0])),
        ]
        self.coarse = CoarseModule(in_channels, out_channel, upsample_size[::-1])
        self.refine = RefineModule(
            out_channel, out_channel, len(in_channels), upsample_size[::-1]
        )

        self.final = nn.Sequential(
            InvertedResidual(out_channel, out_channel, stride=1),
            InvertedResidual(out_channel, out_channel, stride=1),
            nn.Conv2d(
                out_channel,
                cfg.MODEL.NUM_JOINTS,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        # x shape: (N, 3, 128, 128)
        x = self.root(x)  # (N, C, 64, 64)
        x = self.stem(x)  # (N, C, 32, 32)
        x2 = self.stage2(x)  # (N, C, 16, 16)
        x3 = self.stage3(x2)  # (N, C, 8, 8)
        x4 = self.stage4(x3)  # (N, C, 4, 4)

        x_list = self.coarse([x4, x3, x2, x])
        x = self.refine(x_list)
        x = self.final(x)
        return x


def get_pose_net(cfg, is_train, **kwargs):
    model = SFLite(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
        logger.info(f"=> model init!")

    return model
