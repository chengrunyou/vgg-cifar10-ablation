
"""统计各模型参数量、FLOPs 与测试准确率，汇总到 summary.json。"""

import argparse
import json
import os
import sys
import unicodedata

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models import vgg16
from utils.dataset import get_dataloaders
from utils.statistics import count_params, count_flops

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


def load_peak(output_dir, mode):
    """读 history.json，取训练过程中测试准确率最高的一轮，返回 (准确率, 轮次)。"""
    path = os.path.join(output_dir, mode, "history.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        acc_list = json.load(f)["test_acc"]
    best = max(range(len(acc_list)), key=lambda i: acc_list[i])
    return round(acc_list[best], 4), best + 1


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total


def pad(text, width):
    """按显示宽度补空格，中文算两格，用来对齐终端输出。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all"] + list(MODE_CONFIG.keys()))
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    modes = list(MODE_CONFIG.keys()) if args.mode == "all" else [args.mode]
    summary = []

    for mode in modes:
        weight_path = os.path.join(args.output_dir, mode, "best.pth")
        model = vgg16(num_classes=10, **MODE_CONFIG[mode]).to(device)
        n_params = count_params(model)
        n_flops = count_flops(model)
        peak_acc, peak_epoch = load_peak(args.output_dir, mode)
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            acc = evaluate_model(model, test_loader, device)
        else:
            acc = None

        entry = {
            "mode": mode,
            "name": MODE_NAME[mode],
            "params": n_params,
            "flops": n_flops,
            "test_acc": round(acc, 4) if acc is not None else None,
            "peak_acc": peak_acc,
            "peak_epoch": peak_epoch,
        }
        summary.append(entry)
        acc_str = f"{acc:.4f}" if acc is not None else "未训练"
        peak_str = f"{peak_acc:.4f} (第 {peak_epoch} 轮)" if peak_acc is not None else "—"
        print(f"{pad(MODE_NAME[mode], 16)} | 参数量 {n_params:>12,} | "
              f"FLOPs {n_flops:>12,} | 权重复测 {acc_str} | 峰值 {peak_str}")

    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
