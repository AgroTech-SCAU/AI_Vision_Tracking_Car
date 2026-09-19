"""
============================================================
ByteTrack 行人追踪脚本
使用训练好的 YOLO 模型 + ByteTrack 实现多人实时追踪

用法:
    python track.py --source video.mp4
    python track.py --source video.mp4 --output result.mp4
    python track.py --model yolo11n.pt --source 0
============================================================
"""

import argparse
import cv2
import os
import glob
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 运行参数（直接修改这里，优于命令行传参）
# ============================================================
CONF = 0.7        # 检测置信度阈值（0~1，越低检出越多，误检也越多）
IOU = 0.7         # NMS IOU 阈值
IMGSZ = 640       # 推理图像尺寸
DEVICE = "0"      # GPU 编号，CPU 用 "cpu"


def find_latest_model():
    """自动找到 runs/person_detection/ 下最新的 best.pt"""
    pattern = os.path.join(SCRIPT_DIR, "runs", "person_detection", "*", "weights", "best.pt")
    models = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return models[0] if models else os.path.join(SCRIPT_DIR, "yolo11n.pt")


DEFAULT_MODEL = find_latest_model()


def parse_args():
    parser = argparse.ArgumentParser(description="ByteTrack 行人追踪")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO 模型路径")
    parser.add_argument("--source", type=str, required=True, help="输入: 视频路径 / 摄像头编号(0)")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径")
    parser.add_argument("--conf", type=float, default=CONF, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=IOU, help="NMS IOU 阈值")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="推理图像尺寸")
    parser.add_argument("--device", type=str, default=DEVICE, help="设备: 0 或 cpu")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="追踪器配置")
    parser.add_argument("--no-show", action="store_true", help="不显示实时画面")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 加载模型 ----
    model_path = args.model
    if not os.path.exists(model_path):
        alt = os.path.join(SCRIPT_DIR, os.path.basename(model_path))
        if os.path.exists(alt):
            model_path = alt
        else:
            print(f"模型不存在: {args.model}")
            return

    print(f"模型: {model_path}")
    model = YOLO(model_path)

    # ---- 打开视频源 ----
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"无法打开视频源: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {width}x{height}, {fps:.0f}fps, {total}帧")
    print(f"追踪器: {args.tracker}, 置信度: {args.conf}")
    print("-" * 40)

    # ---- 初始化输出视频 ----
    out_writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"输出: {args.output}")

    # ---- 逐帧追踪 ----
    frame_idx = 0
    show = not args.no_show

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            device=args.device,
            imgsz=args.imgsz,
            verbose=False,
        )

        # 绘制追踪结果
        annotated = results[0].plot()

        # 统计人数
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            num_people = len(set(ids))
        else:
            num_people = 0

        if frame_idx % 30 == 0:
            print(f"帧 {frame_idx}/{total}: {num_people} 人")

        # 写入输出视频
        if out_writer:
            out_writer.write(annotated)

        # 显示
        if show:
            cv2.imshow("ByteTrack Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    # ---- 清理 ----
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    print("-" * 40)
    print(f"追踪完成! 共处理 {frame_idx} 帧")
    if args.output:
        print(f"结果已保存到: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
