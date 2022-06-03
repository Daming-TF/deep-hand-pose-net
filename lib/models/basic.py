import logging
import os
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Basic(nn.Module):
    def __init__(self):
        super(Basic, self).__init__()

    def init_weights(self, pretrained=""):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info("=> loading pretrained model {}".format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=True)
        else:
            logger.info("=> init weights from normal distribution")
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ["bias"]:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ["bias"]:
                            nn.init.constant_(m.bias, 0)
