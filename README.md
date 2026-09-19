<h1 align="center">AI Vision Tracking Car</h1>
<h1 align="center">AI视觉跟随小车 </h1>

## **项目介绍** 

🎯 项目名称（暂定名）： AI运动视觉跟随小车 —— 从网球场到客厅的智能陪伴新体验

你是否想过，一台小小的格斗机器人小车，能像忠诚的球童一样，自动追着网球跑、精准拍照、实时分析你的挥拍动作？甚至在你家客厅，它能变身“宠物小跟班”，追着小猫小狗满屋跑，记录它们的可爱瞬间，还能和你互动玩耍？

这就是我们正在筹备的大学生创新创业项目——“智体随行”。它脱胎于的AI+运动视觉科研体系，将实验室里成熟的动作捕捉、姿态估计、运动分析技术，落地成一款可玩、可用、可拓展的智能跟随小车。

🚗 核心功能：

- 🎾 网球运动追踪：小车搭载摄像头+AI视觉模块，自动锁定网球或运动员，边跑边拍，实时输出挥拍轨迹、击球角度、步法分析，帮你科学提升球技。
- 🐶 宠物互动模式：切换至“萌宠跟拍”模式，小车温柔跟随猫狗，自动避障、抓拍趣味瞬间，还能语音互动、播放音乐，成为家庭新宠。
- 🧠 科研+产品双轨并行：项目全程有导师带教，从数据采集、模型训练、实验设计到成果落地，你不仅能产出论文/专利，还能亲手打磨产品原型，积累真实科研与工程经验。



🌟 从网球场到客厅，从科研到产品，从跟随到陪伴——让AI不仅看懂动作，更懂你的生活。

## **功能模块与开发计划**

1.*网球追踪*

实现摄像头对网球实时跟踪，控制云台跟随网球转动

2.*运动员追踪*

实时追踪网球运动员，实现自动锁定、小车移动的同时追踪同一运动员

3.*运动员动作分析*

识别运动员的骨架，分析运动员动作，完成步法分析，同时识别出球拍的运动轨迹，进而识别网球位置，分析击球角度、挥拍轨迹。

4.*宠物识别*

识别猫狗，并给同一猫狗分配固定id，实现小车跟随宠物

5.*宠物动作识别*

识别宠物的骨架，分析宠物动作，在固定姿态下对宠物进行抓拍


## 环境配置

### 1. 打开 Anaconda Prompt

在 Windows 搜索栏中搜索并打开 **Anaconda Prompt**。

### 2. 创建 Conda 环境

```bash
conda create -n <环境名> python=3.10
```

例如：

```bash
conda create -n pytorch-env python=3.10
```

> Python 版本需要根据 PyTorch 的兼容要求选择，通常优先使用 Python 3.10。

### 3. 激活环境

```bash
conda activate <环境名>
```

例如：

```bash
conda activate pytorch-env
```

### 4. 查看显卡支持的 CUDA 版本

在终端中运行：

```bash
nvidia-smi
```

重点查看以下信息：

```text
Driver Version: xxx.xx
CUDA Version: 12.x / 13.x
```

其中：

- `Driver Version` 表示当前安装的 NVIDIA 显卡驱动版本。
- `CUDA Version` 表示当前驱动能够支持的最高 CUDA 版本，并不代表已经安装了对应版本的 CUDA Toolkit。
- 安装 PyTorch 时，应选择 PyTorch 官方提供且不高于该版本的 CUDA 配置。

### 5. 安装 GPU 版本的 PyTorch

打开 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)，根据实际情况选择：

| 选项 | 建议配置 |
|---|---|
| PyTorch Build | Stable |
| Your OS | 根据操作系统选择 |
| Package | Pip |
| Language | Python |
| Compute Platform | 根据 `nvidia-smi` 的结果选择 |

完成选择后，页面底部会生成安装命令。复制 **Run this Command** 中的命令，并在已经激活的 Conda 环境中运行。

例如：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 上述命令仅为示例，请以 PyTorch 官方页面实时生成的命令为准。

### 6. 验证 PyTorch 是否可以使用 GPU

安装完成后运行：

```bash
python
```

然后在 Python 交互环境中依次执行：

```python
import torch

print("PyTorch 版本：", torch.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
print("PyTorch CUDA 版本：", torch.version.cuda)

if torch.cuda.is_available():
    print("显卡型号：", torch.cuda.get_device_name(0))
```

如果输出：

```text
CUDA 是否可用： True
```

则说明 PyTorch GPU 环境配置成功。

输入以下命令退出 Python：

```python
exit()
```

### 7.后续配置

在创建的环境中安装各个模块的依赖、运行代码即可。


