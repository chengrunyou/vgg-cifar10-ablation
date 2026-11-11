"""绘制训练曲线与指标对比图。"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

MODE_NAME = {
    "baseline": "VGG (基线)",
    "dw_separable": "深度可分离卷积",
    "se_attention": "SE 注意力",
    "cbam_attention": "CBAM 注意力",
    "mish": "Mish 激活",
    "gelu": "GELU 激活",
    "groupnorm": "GroupNorm",
}

COLORS = {
    "baseline": "#1f77b4",
    "dw_separable": "#ff7f0e",
    "se_attention": "#2ca02c",
    "cbam_attention": "#d62728",
    "mish": "#9467bd",
    "gelu": "#8c564b",
    "groupnorm": "#e377c2",
}


def load_history(mode):
    path = os.path.join(OUTPUT_DIR, mode, "history.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_curves(modes):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for mode in modes:
        h = load_history(mode)
        if h is None:
            continue
        c = COLORS.get(mode, "black")
        name = MODE_NAME.get(mode, mode)
        axes[0, 0].plot(h["epoch"], h["train_loss"], label=name, color=c)
        axes[0, 1].plot(h["epoch"], h["test_loss"], label=name, color=c)
        axes[1, 0].plot(h["epoch"], h["train_acc"], label=name, color=c)
        axes[1, 1].plot(h["epoch"], h["test_acc"], label=name, color=c)

    axes[0, 0].set_title("训练 Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 1].set_title("测试 Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[1, 0].set_title("训练准确率")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_title("测试准确率")
    axes[1, 1].set_xlabel("Epoch")

    for ax in axes.flat:
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("VGG 基线及各改进版本训练曲线对比", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, "loss_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存 {out}")


def plot_comparison():
    path = os.path.join(OUTPUT_DIR, "summary.json")
    if not os.path.exists(path):
        print("summary.json 不存在，请先运行 evaluate.py")
        return
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    names = [s["name"] for s in summary]
    params = [s["params"] / 1e6 for s in summary]
    flops = [s["flops"] / 1e6 for s in summary]
    accs = [s["test_acc"] if s["test_acc"] is not None else 0 for s in summary]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(names, params, color=[COLORS[s["mode"]] for s in summary])
    axes[0].set_title("参数量 (M)")
    axes[0].set_ylabel("参数量 / 百万")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(names, flops, color=[COLORS[s["mode"]] for s in summary])
    axes[1].set_title("计算量 FLOPs (M)")
    axes[1].set_ylabel("FLOPs / 百万")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(names, accs, color=[COLORS[s["mode"]] for s in summary])
    axes[2].set_title("测试准确率")
    axes[2].set_ylabel("准确率")
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis="x", rotation=45)
    for i, a in enumerate(accs):
        axes[2].text(i, a + 0.01, f"{a:.4f}", ha="center", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("各模型参数量 / 计算量 / 准确率对比", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, "comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存 {out}")


if __name__ == "__main__":
    modes = list(MODE_NAME.keys())
    plot_curves(modes)
    plot_comparison()
    print("绘图完成")
