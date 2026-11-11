"""检查 7 个模型能否正常构建与前向。"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import vgg16
from utils.statistics import count_params, count_flops

MODES = {
    "baseline":       dict(conv_type="standard",    attention="none", activation="relu", norm="bn"),
    "dw_separable":   dict(conv_type="dw_separable", attention="none", activation="relu", norm="bn"),
    "se_attention":   dict(conv_type="standard",    attention="se",   activation="relu", norm="bn"),
    "cbam_attention": dict(conv_type="standard",    attention="cbam", activation="relu", norm="bn"),
    "mish":           dict(conv_type="standard",    attention="none", activation="mish", norm="bn"),
    "gelu":           dict(conv_type="standard",    attention="none", activation="gelu", norm="bn"),
    "groupnorm":      dict(conv_type="standard",    attention="none", activation="relu", norm="gn"),
}

if __name__ == "__main__":
    for mode, cfg in MODES.items():
        m = vgg16(num_classes=10, **cfg)
        p = count_params(m)
        f = count_flops(m)
        out = m(torch.randn(2, 3, 32, 32))
        print(f"{mode:<15} params={p:>12,}  flops={f:>13,}  out_shape={tuple(out.shape)}")
    print("冒烟测试通过")
