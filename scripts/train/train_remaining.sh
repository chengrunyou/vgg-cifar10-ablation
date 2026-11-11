#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
MODES="cbam_attention mish gelu groupnorm"
for m in $MODES; do
  echo "===== 开始训练 $m（20 epoch） ====="
  mkdir -p "outputs/${m}"
  "$PY" -u scripts/train/train.py --mode "$m" --epochs 20 > "outputs/${m}/train.log" 2>&1
  echo "===== $m 训练完成，退出码 $? ====="
done
echo "===== 全部完成 ====="
