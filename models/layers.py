
import torch
import torch.nn as nn

__all__ = ["get_activation", "get_norm", "StandardConv", "DepthwiseSeparableConv", "ConvBlock"]


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "mish":
        return nn.Mish(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"未知激活函数: {name}")


def get_norm(name, num_features):
    name = name.lower()
    if name == "bn":
        return nn.BatchNorm2d(num_features)
    if name == "gn":
        return nn.GroupNorm(num_groups=32, num_channels=num_features)
    raise ValueError(f"未知归一化层: {name}")


class StandardConv(nn.Module):

    def __init__(self, in_channels, out_channels, act="relu", norm="bn", stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu", norm="bn", stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                                   groups=in_channels, bias=False)
        self.norm1 = get_norm(norm, in_channels)
        self.act1 = get_activation(act)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm2 = get_norm(norm, out_channels)
        self.act2 = get_activation(act)

    def forward(self, x):
        x = self.act1(self.norm1(self.depthwise(x)))
        x = self.act2(self.norm2(self.pointwise(x)))
        return x


class ConvBlock(nn.Module):
    @staticmethod
    def build(conv_type, in_channels, out_channels, act="relu", norm="bn", stride=1):
        if conv_type == "standard":
            return StandardConv(in_channels, out_channels, act, norm, stride)
        if conv_type == "dw_separable":
            return DepthwiseSeparableConv(in_channels, out_channels, act, norm, stride)
        raise ValueError(f"未知卷积类型: {conv_type}")
