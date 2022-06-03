import torch.nn as nn
import logging

from lib.models.basic import Basic
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


class MediapipeLiteRegressor(Basic):
    def __init__(self, cfg, **kwargs):
        super(MediapipeLiteRegressor, self).__init__()

        self.num_joints = 21
        self.num_dims = 3

        block = InvertedResidual
        inverted_residual_setting = [
            # expand_ratio, output_channel, kernel_size, stride
            [4, 16, 3, 2],
            [6, 16, 3, 1],
            [6, 24, 5, 2],
            [6, 24, 5, 1],
            [6, 48, 3, 2],
            [6, 48, 3, 1],
            [6, 48, 3, 1],
            [6, 64, 5, 1],
            [6, 64, 5, 1],
            [6, 64, 5, 1],
            [6, 112, 5, 2],
            [6, 112, 5, 1],
            [6, 112, 5, 1],
            [6, 112, 5, 1],
        ]

        input_channel = 3
        hidden_channel = 24
        output_channel = 16
        layers = list()
        layers.extend(
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
        ) in inverted_residual_setting:
            layers.append(
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
        layers.extend(
            [
                ConvBNReLU(input_channel, output_channel, kernel_size=1, stride=1),
                ConvBNReLU(
                    output_channel,
                    output_channel,
                    kernel_size=3,
                    stride=1,
                    groups=output_channel,
                ),
                nn.AdaptiveAvgPool2d(1),
            ]
        )
        self.model = nn.Sequential(*layers)

        # 4-branch output
        # self.handness = nn.Sequential(nn.Linear(output_channel, 1), nn.Sigmoid())
        # self.righthand_pro = nn.Sequential(nn.Linear(output_channel, 1), nn.Sigmoid())
        # self.landmarks = nn.Linear(output_channel, self.num_joints * self.num_dims)
        self.landmarks = nn.Linear(output_channel, self.num_joints * 2)
        # self.world_landmarks = nn.Linear(output_channel, self.num_joints * self.num_dims)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        # handness = self.handness(out)
        # righthand_prop = self.righthand_pro(out)
        # landmarks = self.landmarks(out)
        # world_landmarks = self.world_landmarks(out)
        landmarks = self.landmarks(out)
        landmarks = landmarks.view(landmarks.size(0), self.num_joints, 2)

        # return handness, righthand_prop, landmarks, world_landmarks
        return landmarks


def get_pose_net(cfg, is_train, **kwargs):
    model = MediapipeLiteRegressor(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
        logger.info(f"=> model init!")

    return model
