import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from ultralytics import YOLO


class Detector:
    """目标检测算法类：封装检测逻辑，支持与跟踪器协同工作"""

    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.conf = 0.5
        self.iou = 0.5

        # 单类别网球运动员模型的类别编号为 0。
        self.person_class = 0

        # ROI相关参数
        self.use_roi = False
        self.roi_rect = None  # (x1, y1, x2, y2)
        self.roi_mutex = QMutex()

    def _load_model(self, model_path):
        """加载YOLO模型"""
        try:
            return YOLO(model_path)
        except Exception as e:
            raise ValueError(f"模型加载失败: {str(e)}")

    def set_parameters(self, conf, iou):
        """更新检测参数"""
        self.conf = conf
        self.iou = iou

    def set_roi(self, use_roi, roi_rect=None):
        """设置ROI参数"""
        self.roi_mutex.lock()
        try:
            self.use_roi = bool(use_roi and roi_rect is not None)
            self.roi_rect = roi_rect
        finally:
            self.roi_mutex.unlock()

    def _roi_for_frame(self, frame):
        """读取一致的 ROI 状态，并排除超出画面或为空的区域。"""
        self.roi_mutex.lock()
        try:
            use_roi, roi_rect = self.use_roi, self.roi_rect
        finally:
            self.roi_mutex.unlock()
        if not use_roi or roi_rect is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, roi_rect)
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _crop_roi(self, frame):
        roi_rect = self._roi_for_frame(frame)
        if roi_rect is None:
            return frame, (0, 0)
        x1, y1, x2, y2 = roi_rect
        return frame[y1:y2, x1:x2], (x1, y1)

    def detect_raw(self, frame):
        """
        执行原始检测，返回未处理的检测结果（用于跟踪）
        返回: 检测框数组[N,5] (x1,y1,x2,y2,score) 和类别ID数组[N]
        """
        original_frame = frame.copy()
        frame, roi_offset = self._crop_roi(frame)

        # 执行检测
        results = self.model(
            frame,
            classes=[self.person_class],
            conf=self.conf,
            iou=self.iou,
            stream=False
        )

        detections = []
        class_ids = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, scores, classes):
                if cls_id == self.person_class:
                    # 转换坐标到原始图像
                    x1, y1, x2, y2 = box
                    x1 += roi_offset[0]
                    y1 += roi_offset[1]
                    x2 += roi_offset[0]
                    y2 += roi_offset[1]

                    detections.append([x1, y1, x2, y2, score])
                    class_ids.append(cls_id)

        return np.array(detections), np.array(class_ids)

    def track_frame(self, frame):
        """
        使用 YOLO 内置跟踪器进行目标跟踪（支持 ROI 裁剪）
        返回: (processed_frame, tracked_results, person_count)
        """
        original_frame = frame.copy()
        frame, roi_offset = self._crop_roi(frame)

        # 执行跟踪（persist=True 保持 ID 连续）
        results = self.model.track(
            frame,
            classes=[self.person_class],
            conf=self.conf,
            iou=self.iou,
            persist=True,          # 关键：保持跨帧的跟踪 ID
            tracker="bytetrack.yaml",
            stream=False
        )

        tracked_results = []
        person_count = 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [-1] * len(boxes)

            for box, score, cls_id, track_id in zip(boxes, scores, classes, ids):
                # 只保留网球运动员。
                if cls_id != self.person_class:
                    continue
                # 坐标偏移回原始图像
                x1, y1, x2, y2 = box
                x1 += roi_offset[0]
                y1 += roi_offset[1]
                x2 += roi_offset[0]
                y2 += roi_offset[1]
                tracked_results.append([x1, y1, x2, y2, int(track_id), int(cls_id), float(score)])
                person_count += 1

        # 利用已有的绘制函数绘制跟踪结果
        processed_frame, _ = self.draw_tracked_results(original_frame, tracked_results)
        return processed_frame, tracked_results, person_count

    def draw_detections(self, frame, detections, class_ids):
        """绘制检测结果"""
        processed_frame = frame.copy()
        person_count = 0

        for det, cls_id in zip(detections, class_ids):
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if cls_id == self.person_class:
                person_count += 1
                color = (0, 255, 0)  # 绿色：网球运动员
                label = f"player: {score:.2f}"
            else:
                continue

            # 绘制边界框
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            # 绘制标签
            cv2.putText(processed_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制ROI（如果启用）
        roi_rect = self._roi_for_frame(frame)
        if roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            cv2.rectangle(processed_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(processed_frame, "ROI", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return processed_frame, person_count

    def draw_tracked_results(self, frame, tracked_results):
        """绘制跟踪结果（含跟踪ID）"""
        processed_frame = frame.copy()
        person_count = 0

        for track in tracked_results:
            x1, y1, x2, y2, track_id, cls_id, score = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if cls_id == self.person_class:
                person_count += 1
                color = (0, 255, 0)  # 绿色：网球运动员
            else:
                continue

            # 绘制边界框
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            # 绘制带跟踪ID的标签
            cv2.putText(processed_frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制ROI（如果启用）
        roi_rect = self._roi_for_frame(frame)
        if roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            cv2.rectangle(processed_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(processed_frame, "ROI", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return processed_frame, person_count


class InferenceThread(QThread):
    """支持多目标跟踪的推理线程"""
    update_frame_signal = pyqtSignal(object)  # 发送处理后的帧
    update_stats_signal = pyqtSignal(int)  # 发送网球运动员数量
    process_finished_signal = pyqtSignal()
    error_occurred_signal = pyqtSignal(str)
    frame_position_updated = pyqtSignal(int)  # 发送当前帧位置

    def __init__(self, path, detector, is_image=False, use_tracking=False, tracker=None):
        super().__init__()
        self.path = path
        self.detector = detector
        self.is_image = is_image

        # 跟踪相关参数
        self.use_tracking = use_tracking
        self._use_tracking = use_tracking  # 内部存储
        self._tracker = tracker  # 保留但不使用

        # 线程控制（确保paused是公共属性）
        self.running = False
        self._stop_requested = False
        self.paused = False  # 这是公共属性，可被外部访问
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        self.cap = None

        # 视频位置管理
        self.current_frame_pos = 0
        self.total_frames = 0

    def set_parameters(self, conf, iou):
        """更新检测参数（线程安全）"""
        self.mutex.lock()
        if self.detector:
            self.detector.set_parameters(conf, iou)
        self.mutex.unlock()

    def set_roi(self, use_roi, roi_rect=None):
        """设置ROI参数（线程安全）"""
        self.mutex.lock()
        if self.detector:
            self.detector.set_roi(use_roi, roi_rect)
        # 不再需要更新跟踪器
        self.mutex.unlock()

    def set_tracking(self, use_tracking, tracker=None):
        """线程安全地设置跟踪状态和跟踪器（跟踪器参数保留但忽略）"""
        self.mutex.lock()
        self._use_tracking = use_tracking
        # 忽略 tracker 参数，不再使用外部跟踪器
        self.mutex.unlock()

    def get_current_position(self):
        """获取当前视频位置"""
        self.mutex.lock()
        pos = self.current_frame_pos
        self.mutex.unlock()
        return pos

    def set_start_position(self, pos):
        """设置视频起始位置"""
        self.mutex.lock()
        self.current_frame_pos = pos
        self.mutex.unlock()

    def run(self):
        try:
            self.mutex.lock()
            self.running = not self._stop_requested
            self.paused = False
            should_run = self.running
            self.mutex.unlock()

            if not should_run:
                return

            if self.is_image:
                self._process_image()
            else:
                self._process_video()

            self.process_finished_signal.emit()

        except Exception as e:
            self.error_occurred_signal.emit(f"处理错误: {str(e)}")
        finally:
            self._cleanup()

    def _process_image(self):
        """处理单张图片"""
        frame = cv2.imread(self.path)
        if frame is None:
            self.error_occurred_signal.emit("无法加载图片文件")
            return

        frame = cv2.resize(frame, (1280, 720))
        self.mutex.lock()
        use_tracking = self._use_tracking
        self.mutex.unlock()

        if use_tracking:
            # 使用 YOLO 内置跟踪
            processed_frame, tracked_results, person_count = self.detector.track_frame(frame)
        else:
            detections, class_ids = self.detector.detect_raw(frame)
            processed_frame, person_count = self.detector.draw_detections(
                frame, detections, class_ids)

        # 发送结果
        self.update_frame_signal.emit(processed_frame)
        self.update_stats_signal.emit(person_count)

    def _process_video(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            self.error_occurred_signal.emit("无法打开视频文件")
            return

        while True:
            # 检查线程状态
            self.mutex.lock()
            while self.paused and self.running:
                self.cond.wait(self.mutex)
            if not self.running:
                self.mutex.unlock()
                break

            # 获取当前跟踪状态
            current_use_tracking = self._use_tracking
            self.mutex.unlock()

            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))

            if current_use_tracking:
                # 直接使用 YOLO 跟踪
                processed_frame, tracked_results, p_count = self.detector.track_frame(frame)
            else:
                detections, class_ids = self.detector.detect_raw(frame)
                processed_frame, p_count = self.detector.draw_detections(
                    frame, detections, class_ids)

            # 发送结果
            self.update_frame_signal.emit(processed_frame)
            self.update_stats_signal.emit(p_count)
            self.msleep(33)

    def _cleanup(self):
        """资源清理"""
        self.mutex.lock()
        if self.cap and not self.is_image and self.cap.isOpened():
            self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.release()
        self.running = False
        self.paused = False
        self.mutex.unlock()

    def pause(self):
        """暂停/恢复线程"""
        self.mutex.lock()
        self.paused = not self.paused
        if not self.paused:
            self.cond.wakeAll()
        self.mutex.unlock()

    def stop(self):
        """请求停止，等待当前帧处理结束并由工作线程清理资源。"""
        self.mutex.lock()
        self._stop_requested = True
        self.running = False
        self.paused = False
        self.cond.wakeAll()
        self.mutex.unlock()

        self.wait()
