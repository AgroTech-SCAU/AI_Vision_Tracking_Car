
# YOLO26 宠物动作检测与视频测试

本项目使用 **Ultralytics YOLO26** 检测图片或视频中的宠物，并将检测框归入六种动作类别。当前代码提供训练、验证集评估和视频批量测试三个入口：

- `train.py`：使用项目数据集训练 YOLO26 检测模型。
- `test.py`：评估验证集精度、统计推理耗时，并保存可视化结果和报告。
- `test_video.py`：逐帧测试单个或多个视频，统计各类检测次数，并可输出标注视频。

类别表示**动作**，不区分猫和狗：`jumping`（跳跃）、`lying`（躺卧）、`walking`（行走）、`standing`（站立）、`running`（奔跑）、`sitting`（坐下）。

## 版本记录

当前版本：v0.1  
历史版本：无

## 模块说明

训练使用 `dataset_yolo26/data.yaml` 和根目录的 `yolo26n.pt`。测试脚本会优先选择 `runs/` 中最近修改的 `best.pt`；没有训练权重时，回退到根目录的 `yolo26n.pt`。

## 目录结构

```text
pet_action/
├── dataset_yolo26/
│   ├── data.yaml                 # 数据集路径与六个动作类别
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── test_video/                   # 待测试视频
├── yolo26n.pt                    # 训练起点和测试回退权重
├── train.py                      # 模型训练
├── test.py                       # 验证集评估
├── test_video.py                 # 视频测试
├── runs/                         # 训练输出
└── video_test_results/           # 视频测试输出
```

## 环境要求

- Python 及与之兼容的 Ultralytics、PyTorch 版本。
- NVIDIA GPU 和对应 CUDA 版 PyTorch 用于默认的 `device=0`；没有 GPU 时，可在测试命令中指定 `--device cpu`，并将 `train.py` 中的 `device` 改为 `"cpu"`。
- `opencv-python`、`numpy`、`PyYAML` 用于评估和视频处理。

## 安装配置

建议创建独立的 Python 环境。以下以 Conda 和 PowerShell 为例：

```powershell
conda create -n pet_action python=3.10 -y
conda activate pet_action
python -m pip install --upgrade pip
```

若使用 NVIDIA GPU，先在 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择适合本机驱动与系统的安装命令。随后安装项目依赖：

```powershell
python -m pip install ultralytics opencv-python numpy pyyaml
```

检查当前环境是否识别 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

使用 `device=0` 时，第二行应输出 `True`。

## 准备数据集

`dataset_yolo26/data.yaml` 当前配置如下：

```yaml
train: images/train
val: images/val
names:
  0: jumping
  1: lying
  2: walking
  3: standing
  4: running
  5: sitting
```

图片和标签按同名文件配对，例如：

```text
dataset_yolo26/images/train/cat_jumping_001.png
dataset_yolo26/labels/train/cat_jumping_001.txt
```

每个标签文件按 YOLO 检测格式填写，每个目标占一行：

```text
class_id center_x center_y width height
```

坐标和框尺寸均按图片宽高归一化到 `0～1`。类别编号必须与 `data.yaml` 中的动作名称一致。

## 训练

确认根目录有 `yolo26n.pt`，且数据集目录结构与 `data.yaml` 一致，然后运行：

```powershell
python train.py
```

主要训练参数写在 `train.py` 中：

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `epochs` | 200 | 最多训练轮数；`patience=50` 可提前停止 |
| `imgsz` | 960 | 训练输入尺寸 |
| `batch` | 8 | 每批图片数 |
| `device` | `0` | 第一张 GPU |
| `workers` | 0 | 数据加载进程数，当前设置适合规避 Windows 共享内存错误 |
| `optimizer` | `auto` | 由 Ultralytics 自动选择优化器 |
| `save_period` | -1 | 不额外保存定期检查点 |

训练结果位于 `runs/train*/`，通常包含 `weights/best.pt`、`weights/last.pt` 和训练曲线。当前脚本设为 `resume=False`，重新执行 `python train.py` 会开始一次新的训练。

## 图片评估

评估验证集并生成报告：

```powershell
python test.py
```

指定模型、设备和推理尺寸：

```powershell
python test.py --model runs/train/weights/best.pt --device 0 --size 1440
```

默认使用最近修改的 `best.pt`，数据配置为 `dataset_yolo26/data.yaml`，输出目录为 `test_results/`。脚本会计算验证集指标，保存 `report.json`、`per_image_times.csv` 和 `visualizations/` 中的检测图片。`runs/train/weights/best.pt` 仅是路径示例；实际训练目录可能带编号，请按生成的目录替换。

## 视频测试

默认测试 `test_video/` 目录下的支持格式视频：

```powershell
python test_video.py
```

只测试一个视频：

```powershell
python test_video.py --video test_video/test1.mp4
```

指定模型并调整阈值，或隔帧采样：

```powershell
python test_video.py --model runs/train/weights/best.pt --conf 0.5 --stride 2
```

只生成统计报告，不保存标注视频：

```powershell
python test_video.py --no-save
```

视频测试默认参数为 `imgsz=1440`、`device=0`、`conf=0.1`、`iou=0.35`、`stride=1`。输出位于 `video_test_results/`：`report.json` 包含各视频及各动作类别的统计，`annotated/` 保存带检测框的视频。统计的是**检测框出现次数**，同一只宠物跨多帧会被重复计数。

## 常见问题

### 找不到模型文件

确认根目录存在 `yolo26n.pt`，或使用 `--model` 指定已有的 `.pt` 文件。自动选择的训练权重来自 `runs/` 下最近修改的 `best.pt`。

### 找不到验证集或图片

检查 `dataset_yolo26/data.yaml` 的 `train`、`val` 路径，确认 `images/val/` 与 `labels/val/` 存在，且图片与标签同名。

### `CUDA unavailable` 或找不到 GPU

检查是否安装了与本机驱动兼容的 CUDA 版 PyTorch。也可以在测试命令中使用 `--device cpu`；训练则需修改 `train.py` 的 `device` 参数。

### Windows 报错 `Couldn't open shared file mapping`，错误码 1455

这是系统内存或页面文件不足时可能出现的错误。当前训练脚本设置 `workers=0`；如果本地改成了更大的值，可以先恢复为 `0`。仍报错时，检查 Windows 页面文件和可用内存，并适当降低 `batch` 或 `imgsz`。若保存权重时发生 `MemoryError`，不要直接假定该次生成的 `last.pt` 可用。

## 许可证

项目使用 Ultralytics YOLO；在分发或商用前，请核对 [Ultralytics 许可条款](https://www.ultralytics.com/license)。当前目录未提供独立的项目许可证文件。
