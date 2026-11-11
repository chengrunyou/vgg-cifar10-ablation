import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models import vgg16
from utils.dataset import get_dataloaders, CIFAR10_CLASSES

MODE_CONFIG = {
    "baseline":       dict(conv_type="standard",    attention="none", activation="relu", norm="bn"),
    "dw_separable":   dict(conv_type="dw_separable", attention="none", activation="relu", norm="bn"),
    "se_attention":   dict(conv_type="standard",    attention="se",   activation="relu", norm="bn"),
    "cbam_attention": dict(conv_type="standard",    attention="cbam", activation="relu", norm="bn"),
    "mish":           dict(conv_type="standard",    attention="none", activation="mish", norm="bn"),
    "gelu":           dict(conv_type="standard",    attention="none", activation="gelu", norm="bn"),
    "groupnorm":      dict(conv_type="standard",    attention="none", activation="relu", norm="gn"),
}


def compute_confusion(model, loader, device, num_classes=10):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                cm[t, p] += 1
    return cm


def plot_confusion(cm, mode, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵 - {mode}")

    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    out = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存 {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="baseline", choices=["all"] + list(MODE_CONFIG.keys()))
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
    for mode in modes:
        weight_path = os.path.join(args.output_dir, mode, "best.pth")
        if not os.path.exists(weight_path):
            print(f"[{mode}] 权重不存在，跳过")
            continue
        model = vgg16(num_classes=10, **MODE_CONFIG[mode]).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        cm = compute_confusion(model, test_loader, device)
        out_dir = os.path.join(args.output_dir, mode)
        os.makedirs(out_dir, exist_ok=True)
        plot_confusion(cm, mode, out_dir)
        per_class = cm.diagonal() / cm.sum(axis=1)
        print(f"[{mode}] 各类准确率:")
        for cls, acc in zip(CIFAR10_CLASSES, per_class):
            print(f"    {cls:<10} {acc:.4f}")


if __name__ == "__main__":
    main()
