#!/usr/bin/env python3
"""
宠物动作识别 - 视频测试脚本

功能：
  1. 对目录下的视频逐帧推理，统计每个动作类别的检测次数
  2. 计算推理 FPS、每帧平均检测框数
  3. 输出标注后的视频（框 + 类别 + 置信度）
  4. 生成 JSON 测试报告

用法：
  python test_video.py                          # 自动选最新模型、测试目录下所有 mp4
  python test_video.py --model runs/.../best.pt --video test_video/test1.mp4
  python test_video.py --conf 0.5 --stride 2    # 提阈值 / 隔帧采样快速测试
  python test_video.py --no-save                # 只统计，不输出标注视频
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def project_path(path):
    """将命令行中的相对路径统一按项目根目录解析。"""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="宠物动作识别视频测试")
    parser.add_argument("--model", type=str, default="",
                        help="模型权重路径（.pt），默认自动查找最新 best.pt")
    parser.add_argument("--video", type=str, default="",
                        help="单个视频路径；默认测试目录下所有 *.mp4")
    parser.add_argument("--video-dir", type=str, default="test_video",
                        help="视频所在目录（默认 test_video）")
    parser.add_argument("--imgsz", type=int, default=1440,
                        help="推理图像尺寸（默认 960，与训练一致）")
    parser.add_argument("--device", type=str, default="0",
                        help="推理设备：0 (GPU) / cpu")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.35,
                        help="NMS IoU 阈值")
    parser.add_argument("--stride", type=int, default=1,
                        help="隔帧采样（1=逐帧，2=每2帧测1帧），用于快速测试")
    parser.add_argument("--output", type=str, default="video_test_results",
                        help="结果输出目录")
    parser.add_argument("--no-save", action="store_true",
                        help="不输出标注视频，只做统计")
    return parser.parse_args()


def find_best_model():
    """自动查找最新训练的 best.pt"""
    candidates = sorted(
        (PROJECT_ROOT / "runs").rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = PROJECT_ROOT / "yolo26n.pt"
    if fallback.exists():
        return fallback
    sys.exit("❌ 未找到模型文件，请用 --model 指定路径")


def find_videos(video_dir):
    """查找目录下的所有视频"""
    if not video_dir.is_dir():
        return []
    return sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def format_fps(fps):
    return f"{fps:.2f}" if fps else "N/A"


def process_video(model, video_path, output_dir, args):
    """
    逐帧推理单个视频，返回统计字典。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠ 无法打开视频: {video_path}")
        return None

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出视频写入器
    writer = None
    if not args.no_save:
        out_path = Path(output_dir) / "annotated" / video_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (width, height))
        if not writer.isOpened():
            print(f"  ⚠ 无法创建输出视频（可能缺少编码器），改为不保存视频")
            writer = None

    class_counter = Counter()   # 类别 -> 出现次数（按框统计）
    box_count = []              # 每帧检测框数
    infer_times = []            # 每帧推理耗时(ms)
    frame_idx = 0
    processed = 0

    print(f"  源分辨率: {width}×{height}  源帧率: {format_fps(fps_src)}  "
          f"总帧数: {total_frames}")

    t_start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 隔帧采样
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter()
        results = model.predict(
            frame, imgsz=args.imgsz, device=args.device,
            conf=args.conf, iou=args.iou, verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_times.append(infer_ms)

        det = results[0]
        n = 0
        if det.boxes is not None:
            for box in det.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                class_counter[cls_name] += 1
                n += 1
        box_count.append(n)

        # 写标注帧
        if writer is not None:
            annotated = det.plot()
            writer.write(annotated)

        processed += 1
        frame_idx += 1

    wall_time = time.perf_counter() - t_start
    cap.release()
    if writer is not None:
        writer.release()

    if processed == 0:
        print("  ⚠ 视频无有效帧，跳过")
        return None

    mean_infer = float(np.mean(infer_times)) if infer_times else 0.0
    total_boxes = sum(class_counter.values())

    return {
        "video": video_path.name,
        "resolution": f"{width}×{height}",
        "fps_source": round(fps_src, 2) if fps_src else None,
        "total_frames": total_frames,
        "processed_frames": processed,
        "stride": args.stride,
        "mean_inference_ms": round(mean_infer, 2),
        "inference_fps": round(1000 / mean_infer, 1) if mean_infer > 0 else 0,
        "processing_fps": round(processed / wall_time, 1) if wall_time > 0 else 0,
        "total_detections": total_boxes,
        "avg_boxes_per_frame": round(total_boxes / processed, 3),
        "per_class_counts": dict(class_counter.most_common()),
        "dominant_action": class_counter.most_common(1)[0][0] if class_counter else None,
        "wall_time_s": round(wall_time, 2),
    }


def print_video_report(stat):
    print("  " + "-" * 58)
    print(f"  视频: {stat['video']}   ({stat['resolution']})")
    print(f"  帧数: 处理 {stat['processed_frames']}/{stat['total_frames']} "
          f"(stride={stat['stride']})  耗时 {stat['wall_time_s']}s")
    print(f"  速度: 平均推理 {stat['mean_inference_ms']} ms/帧  "
          f"= {stat['inference_fps']} FPS (推理) / {stat['processing_fps']} FPS (含读写)")
    print(f"  检测: 共 {stat['total_detections']} 框，"
          f"平均 {stat['avg_boxes_per_frame']} 框/帧")
    print(f"  主要动作: {stat['dominant_action']}")
    print("  各类别检测次数:")
    for name, cnt in sorted(stat["per_class_counts"].items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / max(stat["total_detections"], 1) * 30)
        print(f"    {name:<10} {cnt:>6}  {bar}")


def main():
    args = parse_args()
    if args.stride < 1:
        sys.exit("❌ --stride 必须大于等于 1")

    output_dir = project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 模型
    model_path = project_path(args.model) if args.model else find_best_model()
    if not model_path.exists():
        sys.exit(f"❌ 模型文件不存在: {model_path}")
    print(f"[模型] {model_path}")
    model = YOLO(str(model_path))
    print(f"[类别] {model.names}\n")

    # 2. 视频列表
    if args.video:
        videos = [project_path(args.video)]
    else:
        videos = find_videos(project_path(args.video_dir))
    if not videos:
        sys.exit(f"❌ 未找到视频（目录: {args.video_dir}）")
    print(f"[视频] 共 {len(videos)} 个:\n" + "\n".join(f"  - {v}" for v in videos) + "\n")

    # 3. 逐视频测试
    all_stats = []
    print("=" * 60)
    for i, vid in enumerate(videos):
        print(f"[{i + 1}/{len(videos)}] 测试: {vid.name}")
        stat = process_video(model, vid, output_dir, args)
        if stat:
            print_video_report(stat)
            all_stats.append(stat)
        print()
    print("=" * 60)

    # 4. 汇总
    if not all_stats:
        sys.exit("❌ 无任何视频成功测试")

    total_det = sum(s["total_detections"] for s in all_stats)
    overall = Counter()
    for s in all_stats:
        overall.update(s["per_class_counts"])

    print("\n                宠物动作识别 视频测试汇总")
    print("=" * 60)
    print(f"  模型: {model_path}")
    print(f"  设备: {args.device}   imgsz: {args.imgsz}   "
          f"conf: {args.conf}   iou: {args.iou}")
    print(f"  视频数: {len(all_stats)}   总检测框: {total_det}")
    all_infer = [s["mean_inference_ms"] for s in all_stats]
    print(f"  平均推理: {np.mean(all_infer):.2f} ms/帧  "
          f"(≈ {1000 / np.mean(all_infer):.1f} FPS)")
    print("-" * 60)
    print("  各类别总检测次数:")
    for name, cnt in overall.most_common():
        pct = cnt / total_det * 100 if total_det else 0
        print(f"    {name:<10} {cnt:>6}  ({pct:.1f}%)")
    print("=" * 60)

    # 5. 保存 JSON 报告
    report = {
        "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型路径": str(model_path),
        "设备": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "视频": all_stats,
        "汇总": dict(overall.most_common()),
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告已保存: {report_path}")
    if not args.no_save:
        print(f"📁 标注视频保存在: {Path(output_dir) / 'annotated'}")


if __name__ == "__main__":
    main()
