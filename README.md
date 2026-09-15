# BGFD / ChangeCLIP 使用说明

本文件整理 ChangeCLIP 仓库中 BGFD（Building-aware Global Feature Distillation）变化检测项目的训练、测试入口和配套文件。命令均从仓库根目录执行，并只使用仓库相对路径。本文只覆盖以下数据集：

- LEVIR
- S2Looking
- DSIFN
- LEVIR-plus

本文仅讨论上述四个数据集。

## 目录结构

```text
ChangeCLIP/
├── configs/
│   ├── 0cd_ce/                  # 基础 ChangeCLIP 配置
│   ├── xy_test/                 # 本项目实验配置
│   └── changeclip_c2d_distri/   # 按数据集整理的配置
├── tools/
│   ├── train_changeclip.py      # MMEngine 训练入口
│   ├── test.py                  # MMEngine 测试入口
│   ├── dist_train.sh            # 分布式训练封装
│   ├── dist_test.sh             # 分布式测试封装
│   ├── train_dsifn.sh           # DSIFN 训练示例
│   ├── test_*.sh                # 数据集测试封装
│   └── clip_inference.py        # CLIP 辅助推理/伪标签入口
├── mmseg/                       # 模型、数据集和 runner 实现
├── pretrained/                  # CLIP/骨干预训练参数
├── pretrained/ChangeCLIP_best_weights/ # 已保存的 ChangeCLIP 权重
├── work_dirs/                   # 训练日志、配置副本、checkpoint 和测试输出
└── requirements*.txt            # 依赖清单
```

## 环境安装

建议在仓库外创建独立 Python 环境，然后在仓库根目录执行：

```bash
pip install -r requirements/runtime.txt
pip install -e .
```

根据硬件安装匹配版本的 PyTorch。训练和测试入口依赖 MMEngine、MMSegmentation、MMCV（如配置所需）及 CLIP 预训练权重。

## 数据目录约定

数据根目录通过配置文件中的 `data_root` 或运行时 `--cfg-options` 指定。不要把本机绝对路径提交到仓库；推荐在仓库外准备数据，并以相对路径或命令行覆盖配置：

```text
datasets/
├── LEVIR/
├── S2Looking/
├── DSIFN/
└── LEVIR-plus/
```

各数据集应提供配置所需的 train/val/test 图像对和标签目录。实际目录字段以对应配置文件为准。

## 训练入口

通用入口：

```bash
python tools/train_changeclip.py <config> --work-dir <work_dir>
```

分布式封装：

```bash
bash tools/dist_train.sh <config> <gpu_num> --work-dir <work_dir>
```

数据集配置建议：

```text
LEVIR       configs/0cd_ce/changeclip_levir.py
S2Looking   configs/changeclip_c2d_distri/S2Looking/changeclip_levir_spatial_1d_gauss_w_loss_only_8_0.00025_bs20.py
DSIFN       configs/xy_test/dsifn/changeclip_dsifn_train.py
LEVIR-plus  configs/0cd_ce/changeclip_levirplus.py
```

示例（相对路径）：

```bash
bash tools/dist_train.sh configs/0cd_ce/changeclip_levir.py 1 \
  --work-dir work_dirs/bgfd_levir

bash tools/dist_train.sh configs/changeclip_c2d_distri/S2Looking/changeclip_levir_spatial_1d_gauss_w_loss_only_8_0.00025_bs20.py 1 \
  --work-dir work_dirs/bgfd_s2looking

bash tools/dist_train.sh configs/xy_test/dsifn/changeclip_dsifn_train.py 1 \
  --work-dir work_dirs/bgfd_dsifn

bash tools/dist_train.sh configs/0cd_ce/changeclip_levirplus.py 1 \
  --work-dir work_dirs/bgfd_levirplus
```

## 测试入口

通用测试入口：

```bash
python tools/test.py <config> <checkpoint> --work-dir <work_dir> --out <prediction_dir>
```

分布式测试：

```bash
bash tools/dist_test.sh <config> <checkpoint> <gpu_num> --out <prediction_dir>
```

测试完成后，可使用仓库内指标脚本计算结果：

```bash
python tools/general/metric.py \
  --pppred <prediction_dir> \
  --gggt <ground_truth_dir>
```

已有数据集封装入口：

```text
tools/test_levir.sh
tools/test_s2looking.sh
tools/test_dsifn.sh
tools/test_levirplus.sh
```

这些脚本中的历史路径是旧实验环境遗留配置。发布或迁移时，应将数据根目录替换为当前机器上的相对路径/环境变量；不要直接提交服务器绝对路径。

## 权重位置

仓库内已保存的 ChangeCLIP 权重按数据集位于：

```text
pretrained/ChangeCLIP_best_weights/changeclip_levir/
pretrained/ChangeCLIP_best_weights/changeclip_levir_vit/
pretrained/ChangeCLIP_best_weights/changeclip_levirplus/
pretrained/ChangeCLIP_best_weights/changeclip_levirplus_vit/
```

DSIFN 和 S2Looking 的训练产物通常位于：

```text
work_dirs/changeclip_dsifn_train/
work_dirs/changeclip_dsifn_test/
work_dirs/<s2looking_experiment>/
```

使用某个实验时，应优先选择其目录内的 `best_mIoU_iter_*.pth`；若不存在，再按日志记录选择 `iter_*.pth`。不要把临时 smoke test 或失败重试权重当作正式结果。

## 输出与指标

训练输出通常包含：

- `*.log`：训练日志；
- 配置副本：用于复现实际参数；
- `iter_*.pth`、`best_mIoU_iter_*.pth`：checkpoint；
- `test_result/` 或自定义 prediction 目录：预测 mask；
- `metric.py` 输出：F1、Precision、Recall、IoU、Kappa、OA（具体是否有 OA 取决于评测脚本）。

建议每次实验使用独立 `work_dirs/<experiment_name>/`，避免覆盖已有结果。


## 相关文件

- `tools/train_changeclip.py`：训练主入口；
- `tools/test.py`：测试主入口；
- `tools/dist_train.sh`、`tools/dist_test.sh`：分布式封装；
- `tools/general/metric.py`：预测指标计算；
- `configs/0cd_ce/`、`configs/xy_test/`、`configs/changeclip_c2d_distri/`：四个数据集的配置集合；
- `pretrained/`：CLIP 和骨干预训练参数；
- `work_dirs/`：实验日志和权重输出。
