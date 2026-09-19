import sys
import math
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QGridLayout, QGroupBox, QSlider, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from dt_backend import Detector, InferenceThread
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ROIDisplayLabel(QLabel):
    """带ROI绘制功能的显示标签"""
    roi_selected = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.roi_rect = None
        self.temp_pixmap = None
        self.base_pixmap = None
        self.draw_mode = False
        self.original_size = (0, 0)
        self.displayed_rect = None

    def setPixmap(self, pixmap):
        # 保存未缩放画面，绘制和鼠标换算使用同一个显示矩形。
        self.base_pixmap = pixmap
        self.original_size = (pixmap.width(), pixmap.height())
        self.setText("")
        self._refresh_display_geometry()

    def _refresh_display_geometry(self):
        if self.base_pixmap is None or self.base_pixmap.isNull():
            self.displayed_rect = None
            return
        area = self.contentsRect()
        if area.width() <= 0 or area.height() <= 0:
            self.displayed_rect = None
            return
        self.temp_pixmap = self.base_pixmap.scaled(
            area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.displayed_rect = QRect(
            area.x() + (area.width() - self.temp_pixmap.width()) // 2,
            area.y() + (area.height() - self.temp_pixmap.height()) // 2,
            self.temp_pixmap.width(), self.temp_pixmap.height())
        self.update()

    def _clamp_display_point(self, point):
        rect = self.displayed_rect
        return QPoint(
            max(rect.x(), min(point.x(), rect.x() + rect.width())),
            max(rect.y(), min(point.y(), rect.y() + rect.height())))

    def resizeEvent(self, event):
        self.is_drawing = False
        super().resizeEvent(event)
        self._refresh_display_geometry()

    def mousePressEvent(self, event):
        if (self.draw_mode and event.button() == Qt.LeftButton
                and self.displayed_rect is not None
                and self.displayed_rect.contains(event.pos())):
            self.is_drawing = True
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.draw_mode:
            self.end_point = self._clamp_display_point(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing and self.draw_mode and event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.end_point = self._clamp_display_point(event.pos())
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())
            if x2 <= x1 or y2 <= y1:
                self.update()
                return
            rect = self.displayed_rect
            w, h = self.original_size
            self.roi_rect = (
                max(0, min(w, math.floor((x1 - rect.x()) * w / rect.width()))),
                max(0, min(h, math.floor((y1 - rect.y()) * h / rect.height()))),
                max(0, min(w, math.ceil((x2 - rect.x()) * w / rect.width()))),
                max(0, min(h, math.ceil((y2 - rect.y()) * h / rect.height()))))
            self.roi_selected.emit(self.roi_rect)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.displayed_rect is not None and self.temp_pixmap is not None:
            painter = QPainter(self)
            painter.drawPixmap(self.displayed_rect.topLeft(), self.temp_pixmap)
            painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
            if self.draw_mode and self.is_drawing:
                painter.drawRect(
                    min(self.start_point.x(), self.end_point.x()),
                    min(self.start_point.y(), self.end_point.y()),
                    abs(self.end_point.x() - self.start_point.x()),
                    abs(self.end_point.y() - self.start_point.y()))
            elif self.roi_rect is not None:
                x1, y1, x2, y2 = self.roi_rect
                w, h = self.original_size
                rect = self.displayed_rect
                painter.drawRect(
                    rect.x() + round(x1 * rect.width() / w),
                    rect.y() + round(y1 * rect.height() / h),
                    round((x2 - x1) * rect.width() / w),
                    round((y2 - y1) * rect.height() / h))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resource_path = None
        self.is_image = False
        self.model_path = "yolo11n.pt"
        self.inference_thread = None
        self.detector = None
        self.confidence = 0.5
        self.iou_threshold = 0.7

        # 跟踪相关
        self.use_tracking = False

        # ROI相关
        self.use_roi = False
        self.current_roi = None
        self.last_video_position = 0

        self.init_ui()
        self.load_detector()

    def init_ui(self):
        self.setWindowTitle("目标检测与多目标跟踪系统")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 显示区域
        self.display_label = ROIDisplayLabel()
        self.display_label.setText("请加载视频或图片")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #222; color: #aaa; font-size: 14px;")
        self.display_label.roi_selected.connect(self.on_roi_selected)
        main_layout.addWidget(self.display_label, 7)

        # 参数调节
        params_group = QGroupBox("检测参数调节")
        params_layout = QGridLayout()

        self.conf_label = QLabel(f"置信度阈值: {self.confidence:.2f}")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(self.confidence * 100))
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        params_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        params_layout.addWidget(self.conf_slider, 0, 1)
        params_layout.addWidget(self.conf_label, 0, 2)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # 功能控制
        control_layout = QHBoxLayout()

        # ROI控制
        self.enable_roi_btn = QPushButton("启用ROI绘制")
        self.enable_roi_btn.clicked.connect(self.enable_roi_drawing)
        self.clear_roi_btn = QPushButton("清除ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        self.clear_roi_btn.setEnabled(False)
        self.use_roi_checkbox = QPushButton("使用ROI检测")
        self.use_roi_checkbox.setCheckable(True)
        self.use_roi_checkbox.clicked.connect(self.toggle_roi_usage)
        self.use_roi_checkbox.setEnabled(False)

        # ★★★ 跟踪控制 ★★★
        self.tracking_checkbox = QCheckBox("启用多目标跟踪")
        self.tracking_checkbox.stateChanged.connect(self.toggle_tracking)
        self.tracking_status_label = QLabel("跟踪状态: 未启用")
        self.tracking_status_label.setStyleSheet("color: #64748b;")

        control_layout.addWidget(self.enable_roi_btn)
        control_layout.addWidget(self.clear_roi_btn)
        control_layout.addWidget(self.use_roi_checkbox)
        control_layout.addWidget(self.tracking_checkbox)
        control_layout.addWidget(self.tracking_status_label)
        main_layout.addLayout(control_layout)

        # 模型选择和资源加载
        model_layout = QHBoxLayout()
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.clicked.connect(self.select_model)
        self.current_model_label = QLabel(f"当前模型: {os.path.basename(self.model_path)}")
        self.current_model_label.setStyleSheet("color: #333; font-style: italic;")
        model_layout.addWidget(self.select_model_btn)
        model_layout.addWidget(self.current_model_label, 1)
        main_layout.addLayout(model_layout)

        load_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("加载图片")
        self.load_image_btn.clicked.connect(lambda: self.load_resource(True))
        self.load_video_btn = QPushButton("加载视频")
        self.load_video_btn.clicked.connect(lambda: self.load_resource(False))
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)

        load_layout.addWidget(self.load_image_btn)
        load_layout.addWidget(self.load_video_btn)
        load_layout.addWidget(self.start_btn)
        load_layout.addWidget(self.pause_btn)
        load_layout.addWidget(self.stop_btn)
        main_layout.addLayout(load_layout)

        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(main_widget)

    # ---------- 槽函数 ----------
    def on_conf_changed(self):
        self.confidence = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"置信度阈值: {self.confidence:.2f}")
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.set_parameters(self.confidence, self.iou_threshold)
        elif self.detector:
            self.detector.set_parameters(self.confidence, self.iou_threshold)

    def select_model(self):
        self.stop_processing()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pt);;所有文件 (*)"
        )
        if file_path:
            self.model_path = file_path
            self.current_model_label.setText(f"当前模型: {os.path.basename(file_path)}")
            self.load_detector()

    def load_detector(self):
        try:
            self.status_label.setText(f"正在加载模型: {os.path.basename(self.model_path)}...")
            self.detector = Detector(self.model_path)
            self.sync_roi()
            self.status_label.setText(f"模型加载成功: {os.path.basename(self.model_path)}")
        except Exception as e:
            self.detector = None
            self.status_label.setText(f"模型加载失败: {str(e)}")
            QMessageBox.critical(self, "模型错误", f"无法加载模型: {str(e)}")

    def on_roi_selected(self, roi_rect):
        self.current_roi = roi_rect
        self.clear_roi_btn.setEnabled(True)
        self.use_roi_checkbox.setEnabled(True)
        self.sync_roi()

    def sync_roi(self):
        """运行中通过线程更新，未运行时直接同步检测器。"""
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.set_roi(self.use_roi, self.current_roi)
        elif self.detector:
            self.detector.set_roi(self.use_roi, self.current_roi)

    def toggle_roi_usage(self, checked):
        self.use_roi = checked
        self.sync_roi()

    def enable_roi_drawing(self):
        self.display_label.draw_mode = not self.display_label.draw_mode
        self.display_label.is_drawing = False
        self.display_label.update()
        self.enable_roi_btn.setText("禁用ROI绘制" if self.display_label.draw_mode else "启用ROI绘制")

    def clear_roi(self):
        self.current_roi = None
        self.display_label.roi_rect = None
        self.display_label.is_drawing = False
        self.display_label.update()
        self.clear_roi_btn.setEnabled(False)
        self.use_roi_checkbox.setChecked(False)
        self.use_roi = False
        self.sync_roi()

    def load_resource(self, is_image):
        self.stop_processing()
        file_filter = "图片文件 (*.jpg *.jpeg *.png *.bmp)" if is_image else "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            if is_image:
                frame = cv2.imread(file_path)
            else:
                cap = cv2.VideoCapture(file_path)
                try:
                    ret, frame = cap.read()
                    if not ret:
                        frame = None
                finally:
                    cap.release()
            if frame is None:
                QMessageBox.warning(self, "资源错误", "无法读取图片或视频首帧")
                return
            self.is_image = is_image
            self.resource_path = file_path
            self.clear_roi()
            self.status_label.setText(f"已加载: {os.path.basename(file_path)}")
            # 与后端使用相同的处理尺寸，开始前即可框选网球场区域。
            self.update_display(cv2.resize(frame, (1280, 720)))

    # ★★★ 跟踪切换 ★★★
    def toggle_tracking(self, state):
        self.use_tracking = (state == Qt.Checked)
        self.tracking_status_label.setText(f"跟踪状态: {'已启用' if self.use_tracking else '未启用'}")
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.set_tracking(self.use_tracking, None)
            self.status_label.setText(f"多目标跟踪{'已启用' if self.use_tracking else '已禁用'}")
        else:
            self.status_label.setText(f"多目标跟踪{'已启用' if self.use_tracking else '已禁用'}（需重新开始处理生效）")

    def start_processing(self):
        if not self.resource_path or not self.detector:
            QMessageBox.warning(self, "错误", "请先加载模型和资源")
            return

        self.stop_processing()
        self.inference_thread = InferenceThread(
            self.resource_path, self.detector, self.is_image
        )
        self.inference_thread.set_tracking(self.use_tracking, None)
        self.inference_thread.set_roi(self.use_roi, self.current_roi)

        self.inference_thread.update_frame_signal.connect(self.update_display)
        self.inference_thread.update_stats_signal.connect(self.update_stats)
        self.inference_thread.process_finished_signal.connect(self.process_finished)
        self.inference_thread.error_occurred_signal.connect(self.show_error)
        self.inference_thread.frame_position_updated.connect(self.update_frame_position)

        self.inference_thread.start()
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(not self.is_image)
        self.stop_btn.setEnabled(True)

        mode = "跟踪模式" if self.use_tracking else "检测模式"
        self.status_label.setText(f"开始处理 - {mode}")

    def update_display(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qimg))

    def update_stats(self, person_count):
        self.status_label.setText(f"网球运动员: {person_count}")

    def pause_processing(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.pause()
            self.pause_btn.setText("恢复" if self.inference_thread.paused else "暂停")

    def stop_processing(self):
        if self.inference_thread and (self.inference_thread.isRunning() or self.inference_thread.paused):
            self.inference_thread.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.stop_btn.setEnabled(False)

    def process_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("处理完成")

    def show_error(self, message):
        self.status_label.setText(f"错误: {message}")
        QMessageBox.warning(self, "处理错误", message)

    def update_frame_position(self, pos):
        self.current_frame_pos = pos

    def closeEvent(self, event):
        self.stop_processing()
        super().closeEvent(event)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
