
# YOLO + ByteTrack 网球运动员检测与跟踪

本项目使用 **YOLO11** 检测网球运动员，并使用 **ByteTrack** 为视频中连续出现的运动员分配稳定 ID。

项目提供三种使用方式：

- `train.py`：训练单类别网球运动员检测模型。
- `track.py`：命令行检测和 ByteTrack 跟踪视频或摄像头。
- `main.py`：带 ROI（感兴趣区域）选择、检测阈值调节和跟踪开关的桌面界面。

> 当前模型和数据集均为单类别任务：类别 `0` 表示网球运动员。

## 版本记录
当前版本：v0.1\
历史版本：无

## 目录结构

```text
yolo_bytetrack/
├── datasets/
│   ├── data.yaml                 # 数据集配置
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── test_video/                   # 示例视频（可选）
├── yolo11n.pt                    # YOLO11n 预训练权重
├── train.py                      # 训练入口
├── track.py                      # 命令行跟踪入口
├── main.py                       # PyQt5 图形界面
└── dt_backend.py                 # 检测、跟踪、ROI 与后台线程逻辑
```

## 环境要求

- Windows 10/11
- NVIDIA GPU（推荐；CPU 也可运行但速度很慢）
- NVIDIA 驱动与对应 CUDA 版 PyTorch
- Conda / Miniconda
- Python 3.10

训练日志对应的环境为：Python 3.10、Ultralytics 8.4.100、PyTorch 2.13.0 + CUDA 13.0，以及 NVIDIA GeForce RTX 5060 Laptop GPU。其他兼容版本也可以使用，但建议先按下面步骤创建独立环境。

## 安装配置

在 PowerShell 中执行：

```powershell
conda create -n yolo_bytetrack python=3.10 -y
conda activate yolo_bytetrack
python -m pip install --upgrade pip
```

安装支持 NVIDIA GPU 的 PyTorch。请优先根据 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与你显卡驱动匹配的 CUDA 命令。例如当前项目日志所用 CUDA 13.0 轮子源为：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

如果该 CUDA 源没有适合你系统的轮子，请使用官网为你系统生成的命令；不要同时安装 CPU 版和 CUDA 版 PyTorch。

安装项目依赖：

```powershell
pip install ultralytics==8.4.100 opencv-python PyQt5 numpy
```

检查 GPU 是否被 PyTorch 识别：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

输出中的第二行应为 `True`。若为 `False`，请重新按 PyTorch 官网命令安装与本机驱动匹配的 CUDA 版本。

## 准备数据集

每张图片应有同名的 YOLO 格式标签文件：

```text
datasets/images/train/00001.jpg
datasets/labels/train/00001.txt
```

标签每一行格式如下：

```text
class_id center_x center_y width height
```

坐标均为相对于图像宽高归一化后的 `0~1` 数值。网球运动员为唯一类别，因此每行第一个值必须是 `0`：

```text
0 0.445498 0.581350 0.127459 0.544113
```

编辑 `datasets/data.yaml`，**将 `path` 改为你克隆项目后的实际数据集目录**。Windows 下建议使用正斜杠：

```yaml
path: E:/AI_PMVT/yolo_bytetrack/datasets

train: images/train
val: images/val

nc: 1
names:
  0: person
```

`person` 是现有预训练权重中的类别名；在本项目中它的业务含义是网球运动员，类别编号仍为 `0`。

`path` 下必须存在：

```text
images/train/    labels/train/
images/val/      labels/val/
```

## 训练

确保根目录存在 `yolo11n.pt`。如果没有，可从 [Ultralytics YOLO11 发布页](https://github.com/ultralytics/assets/releases) 下载 `yolo11n.pt` 并放到项目根目录。

运行训练：

```powershell
python train.py
```

默认训练配置位于 `train.py`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `epochs` | 50 | 训练轮数 |
| `imgsz` | 640 | 输入图像尺寸 |
| `batch` | 8 | 每批图像数 |
| `device` | `0` | 第一张 NVIDIA GPU |
| `optimizer` | `AdamW` | 优化器 |
| `workers` | 0 | Windows 下避免多进程数据加载与权重保存冲突 |
| `amp` | `False` | 关闭混合精度训练 |

训练输出保存到：

```text
runs/person_detection/<时间戳>/
├── weights/best.pt       # 验证集指标最佳的模型
├── weights/last.pt       # 最后一轮模型
├── results.csv           # 各轮损失与指标
└── results.png           # 训练曲线
```

若训练在保存权重时出现 `ValueError: I/O operation on closed file`，请确认 `workers=0` 没有被改动后重新训练。这个设置会降低数据读取并行度，但可规避 Windows 上已报告的 PyTorch 多进程/序列化问题。

## 命令行跟踪

`track.py` 必须提供 `--source`。默认会自动在 `runs/person_detection` 中选择最新的 `best.pt`；没有找到时使用根目录的 `yolo11n.pt`。

处理示例视频并保存结果：

```powershell
python track.py --source test_video/test.mp4 --output runs/result.mp4
```

明确指定训练后的权重：

```powershell
python track.py --model runs/person_detection/<时间戳>/weights/best.pt --source test_video/test.mp4 --output runs/result.mp4
```

使用 USB 摄像头：

```powershell
python track.py --source 0
```

常用参数：

```powershell
python track.py --source test_video/test.mp4 --conf 0.5 --iou 0.7 --imgsz 640 --device 0
```

批量处理或没有显示器的环境可增加 `--no-show`：

```powershell
python track.py --source test_video/test.mp4 --output runs/result.mp4 --no-show
```

命令行和图形界面均限定 `classes=[0]`，并明确使用 `bytetrack.yaml`。

## 图形界面

运行：

```powershell
python main.py
```

使用步骤：

1. 选择模型文件；默认可选根目录的 `yolo11n.pt` 或训练结果中的 `best.pt`。
2. 加载图片或视频。视频会显示首帧，方便提前选择区域。
3. 如需限定网球场区域，点击“启用ROI绘制”，在画面上框选区域，再点击“使用ROI检测”。
4. 勾选“启用多目标跟踪”可显示每位运动员的 ByteTrack ID。
5. 点击“开始处理”；视频可暂停、恢复或停止。

ROI 的鼠标坐标会自动转换为图像坐标，因此图像缩放、居中显示或窗口大小变化后，选定区域仍会对应正确的球场位置。

## 常见问题

### `track.py: error: the following arguments are required: --source`

没有指定视频或摄像头。示例：

```powershell
python track.py --source test_video/test.mp4
```

### 找不到数据集或训练时使用了旧目录

检查 `datasets/data.yaml` 的 `path`。该路径必须指向本项目中包含 `images` 和 `labels` 两个目录的 `datasets` 文件夹。

### `CUDA unavailable`

说明当前 Python 环境安装的是 CPU 版 PyTorch，或显卡驱动与 PyTorch CUDA 版本不兼容。请按 PyTorch 官方安装页重新安装 CUDA 版 PyTorch。

### 图形界面无法启动

确认激活了项目环境，并重新安装界面依赖：

```powershell
pip install --upgrade PyQt5 opencv-python
```

## 许可证

本项目依赖 Ultralytics YOLO。使用、修改或发布前，请同时确认本项目与 [Ultralytics 许可条款](https://www.ultralytics.com/license) 是否满足你的使用场景。
