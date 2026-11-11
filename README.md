# 基于 VGG 的经典 CNN 架构研究与结构改进

深度学习课程大作业。实验报告单独提交，不在本仓库内。

仓库地址：https://github.com/chengrunyou/vgg-cifar10-ablation

```bash
git clone git@github.com:chengrunyou/vgg-cifar10-ablation.git
```

数据和训练好的权重没有入库（合计约 1.4GB），clone 下来后照「运行 → 2. 准备数据」下载数据；不下载数据也能先跑 `tests/smoke_test.py` 做结构自检。

## 做了什么

在 CIFAR-10 上训练 VGG-16 作为基线，然后把网络的四个核心组件做成可替换开关，每次只改一个，看参数量、计算量和准确率各变多少：

| 改动 | 对照 | 结果 |
|---|---|---|
| 卷积类型 | 标准卷积 → 深度可分离卷积 | 参数量 -87.0%，FLOPs -88.3%，准确率 -8.49pp |
| 注意力 | 无 → SE → CBAM | +0.27pp / +0.47pp |
| 激活函数 | ReLU → Mish / GELU | +0.23pp / +0.63pp |
| 归一化 | BatchNorm → GroupNorm | 不收敛 |

## 环境

- Python 3.11+，PyTorch 2.0+（建议 CUDA 版），NVIDIA GPU
- 本实验实跑环境：torch 2.6.0+cu124、torchvision 0.21.0、numpy 2.4.6、matplotlib 3.11.2、RTX 4060 8GB

```bash
conda create -n cnn python=3.11 -y
conda activate cnn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib
```

装完确认走的是 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 项目结构

```
代码文件/
├── models/                     # 网络定义
│   ├── vgg.py                  # VGG-16，通过四个参数切换组件
│   ├── layers.py               # 标准卷积 / 深度可分离卷积 / 激活 / 归一化
│   └── attention.py            # SE / CBAM
├── utils/
│   ├── dataset.py              # CIFAR-10 加载与数据增强
│   └── statistics.py           # 参数量 / FLOPs
├── scripts/
│   ├── train/train.py          # 训练主程序
│   ├── train/*.sh              # 批量训练与断点续训
│   ├── eval/evaluate.py        # 汇总参数量/FLOPs/准确率
│   ├── eval/confusion_analysis.py  # 混淆矩阵
│   └── visualize/plot_results.py   # 训练曲线与对比图
├── tests/smoke_test.py         # 模型构建自检，不需要数据集
├── requirements.txt
└── .gitignore
```

## 运行

### 1. 自检（不依赖数据）

```bash
python tests/smoke_test.py
```

打印 7 个模型的参数量与 FLOPs。参数量对得上（基线 14,986,698）说明代码和环境都正常。

### 2. 准备数据

CIFAR-10 放项目根目录的 `data/` 下。程序优先读 `data/cifar10_full/cifar10/{train,test}/<类别>/*.png`，没有则回退到 torchvision 自动下载。

官方源在国内很慢，建议走镜像：

```bash
mkdir -p data
python -c "import urllib.request; urllib.request.urlretrieve('https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz','data/cifar10.tgz')"
tar -xzf data/cifar10.tgz -C data/
```

### 3. 训练

```bash
python scripts/train/train.py --mode baseline --epochs 20
```

`--mode` 可选 7 个值：`baseline`、`dw_separable`、`se_attention`、`cbam_attention`、`mish`、`gelu`、`groupnorm`。

产物写入 `outputs/<mode>/`：

| 文件 | 内容 |
|---|---|
| `best.pth` | 测试准确率最高那一轮的权重 |
| `checkpoint.pth` | 最后一轮的完整状态（含优化器），用于续训 |
| `history.json` | 每轮的 loss 与准确率 |
| `train.log` | 训练日志 |

中断后加 `--resume` 从 checkpoint 继续。批量跑 7 个模型：

```bash
bash scripts/train/train_all.sh
```

脚本内部调用 `python`，要指定解释器就先 `export PYTHON=/path/to/python`。

### 4. 评估与绘图

```bash
python scripts/eval/evaluate.py --mode all                 # -> outputs/summary.json
python scripts/visualize/plot_results.py                   # 训练曲线 + 对比图
python scripts/eval/confusion_analysis.py --mode baseline  # 混淆矩阵
```

## 常用参数

四个脚本共用这几个参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data_dir` | `./data` | 数据集位置 |
| `--output_dir` | `./outputs` | 权重与结果位置 |
| `--batch_size` | 128（评估 256）| 批大小 |
| `--num_workers` | 0 | Windows 上建议保持 0 |
| `--resume` | 关闭 | 仅训练脚本，断点续训 |
| `--seed` | 42 | 随机种子 |

数据集在别处时可以这样指过去：

```bash
python scripts/train/train.py --mode gelu --epochs 20 --data_dir D:/data/cifar
```

## 结果

| 模型 | 参数量 | FLOPs | 测试准确率 |
|---|---|---|---|
| VGG（基线） | 14,986,698 | 626,927,616 | 90.37% |
| 深度可分离卷积 | 1,951,595 | 73,469,952 | 81.88% |
| SE 注意力 | 15,218,522 | 627,658,752 | 90.64% |
| CBAM 注意力 | 15,215,300 | 628,945,264 | 90.84% |
| Mish 激活 | 14,986,698 | 626,927,616 | 90.60% |
| GELU 激活 | 14,986,698 | 626,927,616 | 91.00% |
| GroupNorm | 14,986,698 | 626,927,616 | 10.00% |

准确率是各模型 20 轮训练中的最高值，对应 `history.json` 里 `test_acc` 的最大项。`evaluate.py` 会同时打出两列：**权重复测**（加载 `best.pth` 重跑测试集）和**峰值**（读 `history.json`），报告统一采用峰值。参数量与 FLOPs 由 `evaluate.py` 现场统计。

单模型 20 轮在 RTX 4060 上约 20–30 分钟，7 个模型串行约 3 小时。

## 已知问题

**GroupNorm 在 VGG 上不收敛。** 相同配置下 loss 恒为 2.3028（10 分类的随机水平），准确率停在 10%。试过换 Adam、去掉 weight_decay、把 lr 降到 0.001、num_groups 从 32 改成 8，都无效。推测原因是 VGG 没有残差连接，GroupNorm 又失去了 BatchNorm 在 batch 维度的耦合，深层梯度衰减到无法更新。报告中作为负面结论记录。

**深度可分离卷积精度掉 8.49pp。** 这是纯结构替换、没有调整宽度和深度的结果，属于正常的表达力损失，不是实现错误。

## 几个坑

- **PyTorch 装成 CPU 版**：直接 `pip install torch` 会装 CPU 版，训练慢到无法接受。用上面的 `--index-url` 指定 CUDA 版本。
- **Windows 上 num_workers > 0 卡死**：DataLoader 多进程在 Windows 下容易卡在第一个 batch，脚本默认 `num_workers=0`。
- **数据集解压**：`cifar10.tgz` 解压后是 6 万个小 PNG，Windows 上建议用系统 `tar`，Python 的 `tarfile` 解压很慢。
