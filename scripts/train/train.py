

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models import vgg16
from utils.dataset import get_dataloaders
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=list(MODE_CONFIG.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vgg16(num_classes=10, **MODE_CONFIG[args.mode]).to(device)
    print(f"[{args.mode}] 参数量: {count_params(model):,}", flush=True)

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pth")
    history_path = os.path.join(out_dir, "history.json")
    best_path = os.path.join(out_dir, "best.pth")

    start_epoch = 1
    best_acc = 0.0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    if args.resume and os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["test_loss"].append(round(test_loss, 4))
        history["test_acc"].append(round(test_acc, 4))

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | {elapsed:.1f}s", flush=True)


        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
        }, ckpt_path)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
    print(f"[{args.mode}] 训练完成，最佳测试准确率: {best_acc:.4f}", flush=True)
    print(f"[{args.mode}] 权重已保存至 {best_path}", flush=True)


if __name__ == "__main__":
    main()
