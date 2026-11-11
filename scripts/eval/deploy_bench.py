"""测各模型的权重体积与单张图片的推理延迟，供报告的部署测算使用。"""

import argparse
import json
import os
import statistics
import sys
import time
import unicodedata

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models import vgg16
from utils.statistics import count_params

MODE_CONFIG = {
    "baseline":       dict(conv_type="standard",    attention="none", activation="relu", norm="bn"),
    "dw_separable":   dict(conv_type="dw_separable", attention="none", activation="relu", norm="bn"),
    "se_attention":   dict(conv_type="standard",    attention="se",   activation="relu", norm="bn"),
    "cbam_attention": dict(conv_type="standard",    attention="cbam", activation="relu", norm="bn"),
    "mish":           dict(conv_type="standard",    attention="none", activation="mish", norm="bn"),
    "gelu":           dict(conv_type="standard",    attention="none", activation="gelu", norm="bn"),
    "groupnorm":      dict(conv_type="standard",    attention="none", activation="relu", norm="gn"),
}

MODE_NAME = {
    "baseline": "VGG (基线)",
    "dw_separable": "深度可分离卷积",
    "se_attention": "SE 注意力",
    "cbam_attention": "CBAM 注意力",
    "mish": "Mish 激活",
    "gelu": "GELU 激活",
    "groupnorm": "GroupNorm",
}


def pad(text, width):
    """按显示宽度补空格，中文算两格。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - w)


def latency(model, device, iters, warmup=10):
    """单张 32×32 输入的前向延迟，取中位数（毫秒）。"""
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_iters", type=int, default=20)
    parser.add_argument("--gpu_iters", type=int, default=100)
    parser.add_argument("--output_dir", default="./outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"{pad('模型', 16)} | 参数量 | FP32 体积 | GPU 延迟(ms) | CPU 延迟(ms)")

    rows = []
    for mode, cfg in MODE_CONFIG.items():
        model = vgg16(num_classes=10, **cfg).to(device)
        n_params = count_params(model)
        size_mb = n_params * 4 / 1024 / 1024
        gpu_ms = latency(model, device, args.gpu_iters) if device.type == "cuda" else None
        model.to("cpu")
        cpu_ms = latency(model, torch.device("cpu"), args.cpu_iters)
        row = {
            "mode": mode,
            "name": MODE_NAME[mode],
            "params": n_params,
            "size_mb": round(size_mb, 2),
            "gpu_ms": round(gpu_ms, 2) if gpu_ms else None,
            "cpu_ms": round(cpu_ms, 2),
        }
        rows.append(row)
        gpu_str = f"{gpu_ms:12.2f}" if gpu_ms else "            -"
        print(f"{pad(MODE_NAME[mode], 16)} | {n_params:>12,} | {size_mb:7.2f} MB | "
              f"{gpu_str} | {cpu_ms:11.2f}")

    out = os.path.join(args.output_dir, "deploy_bench.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {out}")


if __name__ == "__main__":
    main()
