
import torch
import torch.nn as nn

from .attention import SEBlock, CBAMBlock
from .layers import ConvBlock

__all__ = ["VGG", "vgg16"]

CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
       512, 512, 512, "M", 512, 512, 512, "M"]


class VGG(nn.Module):
    def __init__(self, cfg=CFG, num_classes=10, conv_type="standard",
                 attention="none", activation="relu", norm="bn"):
        super().__init__()
        self.conv_type = conv_type
        self.activation = activation
        self.norm = norm

        self.features = self._make_features(cfg, attention)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _make_features(self, cfg, attention):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(ConvBlock.build(
                    self.conv_type, in_channels, v,
                    act=self.activation, norm=self.norm,
                ))
                if attention == "se":
                    layers.append(SEBlock(v))
                elif attention == "cbam":
                    layers.append(CBAMBlock(v))
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16(**kwargs):
    return VGG(cfg=CFG, **kwargs)
