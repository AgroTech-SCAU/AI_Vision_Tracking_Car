
# YOLO11 猫狗目标检测

本项目使用 **Ultralytics YOLO11** 训练猫狗目标检测模型。当前代码只有训练入口 `pet.py`；目录中保留了一次训练的权重、指标和可视化结果，可用已有权重对新图片或视频进行推理。

## 版本记录

当前版本：v0.1  
历史版本：无

## 模块说明

- `pet.py`：从 `yolo11n.pt` 开始训练，读取 `data/openimages_cat_dog_yolo/data.yaml` 中的数据集配置。
- `runs/weights/best.pt`：当前保存的最佳训练权重，可用于推理。
- `runs/weights/last.pt`：当前保存的最后一轮权重。
- `runs/results.csv`、`runs/results.png`：逐轮训练指标及曲线。
- `runs/` 下的其余图片：标签分布、训练批次、验证预测、混淆矩阵及指标曲线。

当前目录**没有** `data/` 数据集、`data.yaml` 或 `yolo11n.pt` 初始权重。重新训练前需要准备这些文件；使用现有 `best.pt` 推理则不需要训练数据集。

## 目录结构

```text
cat_dog_v0.1/
├── pet.py                       # 训练脚本
├── README.md                    # 项目说明
└── runs/                        # 已保存的一次训练结果
    ├── args.yaml                # 当次训练的完整参数
    ├── results.csv              # 每轮训练与验证指标
    ├── results.png              # 训练曲线
    ├── confusion_matrix.png     # 混淆矩阵
    ├── val_batch*_pred.jpg      # 验证集预测示例
    └── weights/
        ├── best.pt              # 最佳权重
        └── last.pt              # 最后一轮权重
```

## 环境要求

- Python、Ultralytics 及其兼容的 PyTorch 版本。
- `pet.py` 设置 `device=0`，默认使用第一张 NVIDIA GPU；仅有 CPU 时需将该参数改为 `"cpu"`。
- 训练需要数据集及相应的 `data.yaml`；推理只需安装依赖并使用现有权重。

## 安装配置

以下以 Conda 和 PowerShell 为例：

```powershell
conda create -n cat_dog python=3.10 -y
conda activate cat_dog
python -m pip install --upgrade pip
python -m pip install ultralytics
```

若使用 GPU，请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装与本机环境匹配的版本。可用以下命令检查 PyTorch 是否识别 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

使用 `device=0` 时应输出 `True`。

## 准备数据集

训练脚本固定读取 `data/openimages_cat_dog_yolo/data.yaml`。该目录目前不在项目中，需要自行放入 YOLO 检测格式的数据集，或修改 `pet.py` 中的 `data` 路径。目录可按下面的形式组织：

```text
data/openimages_cat_dog_yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`data.yaml` 需要声明训练集、验证集及类别。以下仅为猫狗两类数据的**示例**；实际类别编号和顺序应以自己的标注为准：

```yaml
path: data/openimages_cat_dog_yolo
train: images/train
val: images/val
names:
  0: cat
  1: dog
```

图片与标签文件按同名配对，如 `images/train/example.jpg` 与 `labels/train/example.txt`。每个目标在标签文件中占一行：

```text
class_id center_x center_y width height
```

框中心坐标和宽高按图片尺寸归一化到 `0～1`。若从项目根目录运行脚本，上面示例中的 `path` 应能正确定位数据集；也可以在 `data.yaml` 中使用绝对路径。

## 训练

准备好数据集配置和 `yolo11n.pt` 初始权重后，在项目根目录运行：

```powershell
python pet.py
```

`pet.py` 中的主要参数如下：

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `epochs` | 150 | 最多训练轮数 |
| `patience` | 30 | 验证指标持续未改善时提前停止 |
| `imgsz` | 640 | 输入图片尺寸 |
| `batch` | 8 | 每批图片数 |
| `device` | `0` | 第一张 GPU |
| `mosaic` / `close_mosaic` | `0.5` / `10` | Mosaic 增强概率及最后 10 轮关闭增强 |
| `mixup` | `0.0` | 不使用 MixUp |
| `scale` / `translate` / `fliplr` | `0.45` / `0.1` / `0.5` | 缩放、平移与水平翻转增强 |
| `cos_lr` / `lr0` / `lrf` | `True` / `0.001` / `0.05` | 余弦学习率及其参数 |
| `name` | `openimages_cat_dog` | 训练结果目录名称 |

Ultralytics 通常将新训练结果保存到 `runs/detect/openimages_cat_dog*/`，其中包含 `weights/best.pt` 和 `weights/last.pt`。现有结果单独存放于本项目的 `runs/` 目录，具体训练参数可查看 `runs/args.yaml`。重新运行时若结果目录已存在，Ultralytics 可能给新目录追加编号。

## 使用已有权重推理

在项目根目录执行以下命令，对单张图片进行检测：

```powershell
yolo detect predict model=runs/weights/best.pt source="path/to/image.jpg" save=True
```

将 `source` 替换为本机图片、图片目录或视频路径；预测结果通常保存在 `runs/detect/predict*/`。也可以用 Python 调用：

```python
from ultralytics import YOLO

model = YOLO("runs/weights/best.pt")
model.predict(source="path/to/image.jpg", save=True)
```

权重中的类别名称取决于实际训练数据集。由于当前目录不含原始 `data.yaml`，请通过推理结果或 `model.names` 核对类别编号与名称。

## 常见问题

### 找不到数据集配置

确认 `data/openimages_cat_dog_yolo/data.yaml` 已放入项目，且其中的训练集、验证集路径指向实际图片。若数据集在其他位置，修改 `pet.py` 的 `data` 参数。

### 找不到 `yolo11n.pt`

重新训练需要可访问的 YOLO11n 初始权重。可将文件放在项目根目录，或把 `pet.py` 的模型路径改为已有权重。当前 `runs/weights/best.pt` 可直接用于推理。

### `CUDA unavailable` 或显存不足

检查 GPU 驱动与 PyTorch 是否匹配。没有 GPU 时将 `pet.py` 的 `device=0` 改为 `device="cpu"`；显存不足时可降低 `batch` 或 `imgsz`。

## 许可证

当前目录未提供独立的项目许可证文件。分发或商用模型及代码前，请核对 [Ultralytics 许可条款](https://www.ultralytics.com/license)。
