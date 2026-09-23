from multiprocessing import freeze_support

from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    # 加载当前最佳模型继续微调
    model = YOLO(str(PROJECT_ROOT / "yolo26n.pt"))

    model.train(
        data=str(PROJECT_ROOT / "dataset_yolo26" / "data.yaml"),
            epochs=200,
            imgsz=960,
            batch=8,
            nbs=64,                # 标称 batch size，补偿 batch=1 的 loss 归一化
            name="train",
            patience=50,
            device=0,
            workers=0,
            optimizer="auto",
            cos_lr=True,
            lr0=0.001,            # 微调学习率（之前 0.0000001 太小）
            lrf=0.01,              # 最终 lr = lr0 × lrf = 5e-6
            warmup_epochs=3,       # 预热轮数（之前 10 太多）
            close_mosaic=100,        # 最后 N 轮关闭 mosaic
        
            # ========== 数据增强 ==========
            mosaic=1.0,            # 马赛克增强（小目标检测必开）
            copy_paste=0.15,       # 复制粘贴增强（小目标最有效的数据增强）
            scale=0.7,             # 多尺度缩放
            hsv_h=0.015,
            hsv_s=0.1,
            hsv_v=0.1,
            flipud=0.27,
            fliplr=0.5,
        
            # ========== 正则化 ==========
            dropout=0.1,           # Dropout（之前 0.516 太高）
            label_smoothing=0.0,
        
            # ========== 模型保存 ==========
            save=True,
            save_period=-1,
            exist_ok=False,
            project=str(PROJECT_ROOT / "runs"),
            resume=False,
        
    )

if __name__ == "__main__":
    freeze_support()
    main()