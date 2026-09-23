#!/usr/bin/env python3
"""
荔枝检测模型 - 综合测试脚本
功能：
  1. 模型精度评估（mAP, Precision, Recall, F1 等）
  2. 推理时间统计（预处理、推理、后处理各阶段耗时）
  3. 可视化检测结果并保存
  4. 导出详细测试报告

用法：
  python test.py                          # 自动选择最新训练的模型
  python test.py --model best.pt          # 指定模型路径
  python test.py --model best.pt --size 1440 --device cpu  # 自定义参数
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path):
    """将命令行中的相对路径统一按项目根目录解析。"""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="荔枝检测模型测试")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="模型权重路径（.pt），默认自动选择 runs 下最新的 best.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset_yolo26/data.yaml",
        help="数据集配置文件路径",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1440,
        help="推理图像尺寸（默认 1440）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备：0 (GPU), cpu, 或 mps",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 阈值（默认 0.45）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results",
        help="结果输出目录",
    )
    parser.add_argument(
        "--vis-samples",
        type=int,
        default=10,
        help="可视化样本数量（默认 10）",
    )
    return parser.parse_args()


def find_best_model():
    """自动查找最新的训练模型"""
    candidates = sorted(
        (PROJECT_ROOT / "runs").rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # fallback
    fallback = PROJECT_ROOT / "yolo26n.pt"
    if fallback.exists():
        return fallback
    sys.exit("未找到模型文件，请用 --model 指定路径")


def find_validation_images(data_yaml):
    """按 data.yaml 解析验证集目录，避免在代码中重复写死数据集路径。"""
    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    val_value = config.get("val")
    if not isinstance(val_value, str):
        sys.exit(f"❌ 数据集配置中的 val 路径无效: {data_yaml}")

    root_value = config.get("path")
    dataset_root = Path(root_value).expanduser() if root_value else data_yaml.parent
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    val_img_dir = Path(val_value).expanduser()
    if not val_img_dir.is_absolute():
        val_img_dir = (dataset_root / val_img_dir).resolve()

    if not val_img_dir.exists():
        sys.exit(f"❌ 验证集目录不存在: {val_img_dir}")

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in val_img_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes
    )


def format_time(ms):
    """格式化时间显示"""
    if ms < 0.001:
        return f"{ms * 1_000_000:.1f} µs"
    elif ms < 1:
        return f"{ms * 1000:.1f} µs"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    else:
        return f"{ms / 1000:.2f} s"


def warmup(model, imgsz, device, n=10):
    """预热模型，排除首次推理的初始化开销"""
    print(f"[预热] 运行 {n} 次推理预热...")
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(n):
        _ = model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    print("  预热完成 ✓\n")


def benchmark_inference(model, img_paths, imgsz, device, conf, iou):
    """
    逐张推理并统计各阶段耗时：
      - preprocess: 图像读取 + 缩放 + 转 tensor
      - inference:  模型前向传播
      - postprocess: NMS + 坐标映射
      - total:      整张图从读到结果的总耗时
    返回:
      times: list[dict]  每张图各阶段耗时
      all_detections: list[list]  每张图的检测结果
    """
    print(f"[推理基准] 开始测试 {len(img_paths)} 张验证集图像...\n")
    times = []
    all_detections = []

    for idx, img_path in enumerate(img_paths):
        img_path = str(img_path)

        # ---------- preprocess ----------
        t0 = time.perf_counter()
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  ⚠ 无法读取 {img_path}，跳过")
            continue
        h0, w0 = img_bgr.shape[:2]
        # Ultralytics 内部会做预处理，这里只统计到读取+记录原始尺寸
        t1 = time.perf_counter()

        # ---------- inference ----------
        t2 = time.perf_counter()
        results = model.predict(
            img_bgr,
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        t3 = time.perf_counter()

        # ---------- postprocess ----------
        # predict 内部已完成 NMS，这里提取结果并做坐标映射
        t4 = time.perf_counter()
        dets = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                conf_val = float(box.conf[0])
                dets.append({
                    "class": cls_id,
                    "name": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": [round(x, 1) for x in xyxy],
                })
        t5 = time.perf_counter()

        preprocess_ms = (t1 - t0) * 1000
        inference_ms = (t3 - t2) * 1000
        postprocess_ms = (t5 - t4) * 1000
        total_ms = (t5 - t0) * 1000

        times.append({
            "image": os.path.basename(img_path),
            "resolution": f"{w0}×{h0}",
            "preprocess_ms": round(preprocess_ms, 3),
            "inference_ms":  round(inference_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "total_ms":      round(total_ms, 3),
            "num_detections": len(dets),
        })
        all_detections.append(dets)

        # 每 20 张打印一次进度
        if (idx + 1) % 20 == 0 or idx == len(img_paths) - 1:
            print(f"  进度: {idx + 1}/{len(img_paths)}  |  "
                  f"平均推理时间: {np.mean([t['inference_ms'] for t in times]):.2f} ms")

    print(f"\n推理基准测试完成 ✓\n")
    return times, all_detections


def compute_metrics_direct(model, data_yaml, imgsz, device, conf, iou):
    """
    使用 Ultralytics val 模式直接计算 mAP 等精度指标。
    这是最标准、最准确的精度评估方式。
    """
    print("[精度评估] 调用 model.val() 计算 mAP...\n")
    t_start = time.perf_counter()
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        split="val",
        verbose=True,
    )
    t_end = time.perf_counter()
    eval_time = t_end - t_start
    print(f"\n精度评估完成，耗时 {format_time(eval_time * 1000)} ✓\n")
    return metrics, eval_time


def visualize_samples(model, img_paths, output_dir, imgsz, device, conf, iou, n=10):
    """抽样可视化检测结果"""
    print(f"[可视化] 随机抽取 {n} 张图像进行可视化...")
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(42)
    samples = random.sample(img_paths, min(n, len(img_paths)))

    for idx, img_path in enumerate(samples):
        img_path = str(img_path)
        results = model.predict(img_path, imgsz=imgsz, device=device,
                                conf=conf, iou=iou, verbose=False)
        annotated = results[0].plot()
        out_path = vis_dir / f"det_{idx:02d}_{os.path.basename(img_path)}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  [{idx + 1}/{n}] 保存 → {out_path.name}")

    print(f"可视化完成，结果保存在 {vis_dir.absolute()}\n")


def print_summary(times, metrics, eval_time, output_dir, model_path, args):
    """打印并保存测试报告"""
    # ---- 推理时间统计 ----
    inf_times = [t["inference_ms"] for t in times]
    total_times = [t["total_ms"] for t in times]

    summary = {
        "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型路径": str(model_path),
        "推理设备": args.device,
        "图像尺寸": args.size,
        "置信度阈值": args.conf,
        "IoU阈值": args.iou,
        "测试图像数": len(times),
        # 推理时间
        "推理时间_ms": {
            "平均值": round(np.mean(inf_times), 2),
            "中位数": round(np.median(inf_times), 2),
            "标准差": round(np.std(inf_times), 2),
            "最小值": round(np.min(inf_times), 2),
            "最大值": round(np.max(inf_times), 2),
        },
        "总处理时间_ms": {
            "平均值": round(np.mean(total_times), 2),
            "中位数": round(np.median(total_times), 2),
            "总计": round(np.sum(total_times), 2),
        },
        "FPS": round(1000 / np.mean(inf_times), 1),
        # ---- 精度指标 ----
        "精度评估耗时_s": round(eval_time, 2),
    }

    # 从 metrics 提取核心指标
    if hasattr(metrics, "results_dict") and metrics.results_dict:
        for k, v in metrics.results_dict.items():
            summary[k] = round(float(v), 4) if isinstance(v, (int, float)) else str(v)

    # ---- 打印 ----
    print("=" * 65)
    print("                    荔 枝 检 测 模 型 测 试 报 告")
    print("=" * 65)
    print(f"  模型: {model_path}")
    print(f"  设备: {args.device}    图像尺寸: {args.size}    图像数: {len(times)}")
    print(f"  Conf: {args.conf}    IoU: {args.iou}")
    print("-" * 65)
    print(f"  📊 推理时间 (仅模型前向):")
    print(f"     平均: {summary['推理时间_ms']['平均值']} ms  |  "
          f"中位数: {summary['推理时间_ms']['中位数']} ms")
    print(f"     最小: {summary['推理时间_ms']['最小值']} ms  |  "
          f"最大: {summary['推理时间_ms']['最大值']} ms")
    print(f"     标准差: {summary['推理时间_ms']['标准差']} ms")
    print(f"     FPS (仅推理): {summary['FPS']}")
    print(f"  📊 总处理时间 (读取+推理+NMS):")
    print(f"     平均: {summary['总处理时间_ms']['平均值']} ms")
    print(f"     总计: {summary['总处理时间_ms']['总计']} ms "
          f"({summary['总处理时间_ms']['总计'] / 1000:.1f} s)")
    print("-" * 65)
    print(f"  🎯 精度指标:")
    for k, v in summary.items():
        if k.startswith("metrics/") or k in ("mAP50", "mAP50-95", "precision", "recall", "f1"):
            print(f"     {k}: {v}")
    print("-" * 65)
    print(f"  精度评估耗时: {summary['精度评估耗时_s']} s")
    print(f"  结果保存在: {output_dir}")
    print("=" * 65)

    # ---- 保存 JSON 报告 ----
    report_path = Path(output_dir) / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📄 详细报告已保存: {report_path}")

    # ---- 保存逐张推理时间 CSV ----
    csv_path = Path(output_dir) / "per_image_times.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,resolution,preprocess_ms,inference_ms,postprocess_ms,total_ms,num_detections\n")
        for t in times:
            f.write(f"{t['image']},{t['resolution']},"
                    f"{t['preprocess_ms']},{t['inference_ms']},"
                    f"{t['postprocess_ms']},{t['total_ms']},{t['num_detections']}\n")
    print(f"📄 逐张耗时 CSV: {csv_path}")


def main():
    args = parse_args()
    output_dir = project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 选择模型
    model_path = project_path(args.model) if args.model else find_best_model()
    if not model_path.exists():
        sys.exit(f"❌ 模型文件不存在: {model_path}")
    print(f"[模型] 加载: {model_path}\n")

    # 2. 加载模型
    model = YOLO(str(model_path))

    # 3. 获取验证集图像列表
    data_yaml = project_path(args.data)
    if not data_yaml.is_file():
        sys.exit(f"❌ 数据集配置文件不存在: {data_yaml}")
    img_paths = find_validation_images(data_yaml)
    if not img_paths:
        sys.exit(f"❌ 数据集配置指定的验证集中无图像: {data_yaml}")
    print(f"[数据] 验证集图像数: {len(img_paths)}\n")

    # 4. 预热
    warmup(model, args.size, args.device)

    # 5. 推理时间基准测试
    times, all_detections = benchmark_inference(
        model, img_paths, args.size, args.device, args.conf, args.iou
    )

    # 6. 精度评估（mAP 等）
    metrics, eval_time = compute_metrics_direct(
        model, str(data_yaml), args.size, args.device, args.conf, args.iou
    )

    # 7. 可视化样本
    visualize_samples(model, img_paths, output_dir,
                      args.size, args.device, args.conf, args.iou, args.vis_samples)

    # 8. 汇总报告
    print_summary(times, metrics, eval_time, output_dir, model_path, args)


if __name__ == "__main__":
    main()
