#!/usr/bin/env bash
# 自动续训驱动：反复调用 train.py --resume，直到达到目标 epoch
# 用法: bash scripts/train_auto.sh <mode> <target_epoch>
MODE="$1"
TARGET="$2"
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
mkdir -p "outputs/${MODE}"

while true; do
  # 检查当前 checkpoint 进度
  if [ -f "outputs/${MODE}/checkpoint.pth" ]; then
    CUR=$("$PY" -c "import torch; c=torch.load('outputs/${MODE}/checkpoint.pth', map_location='cpu'); print(c['epoch'])")
  else
    CUR=0
  fi
  if [ "$CUR" -ge "$TARGET" ]; then
    echo "[$MODE] 已达到目标 $TARGET epoch（当前 $CUR），完成"
    break
  fi
  echo "[$MODE] 当前 epoch=$CUR，继续训练到 $TARGET ..."
  "$PY" -u scripts/train/train.py --mode "$MODE" --epochs "$TARGET" --resume >> "outputs/${MODE}/train.log" 2>&1
  echo "[$MODE] 本次训练中断/结束，退出码 $?"
  sleep 2
done
echo "[$MODE] 全部完成"
