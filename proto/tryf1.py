# Final_run_pyside6_glass_v2.py
# Neo-Glassmorphism (Strong Frosted Glass) + Soft Purple Glow (Mode 1)
# Integrated high-FPS detection (no disk I/O). PySide6 + OpenCV.
#
# Requirements:
#   pip install opencv-python numpy PySide6
#
# Place next to config_files/ and run.

import sys
import os
import time
import math
from collections import defaultdict
import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QLinearGradient, QBrush, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy, QSpacerItem, QMessageBox, QFileDialog,
    QGraphicsBlurEffect, QGraphicsDropShadowEffect
)

# ---------------- Config ----------------
THRESH = 0.6
NMS_THRESH = 0.4
ROI_LINE_REL = 0.66
ABANDONED_SECONDS = 8
ABANDONED_DISTANCE = 80
OBJECT_CLASSES_TO_TRACK = {"backpack", "handbag", "suitcase", "bottle", "bag"}

MOTION_MIN_AREA = 800
MOTION_PAUSE_SECONDS = 10.0

FRAME_W = 640
FRAME_H = 480
TIMER_INTERVAL_MS = 33  # ~30 FPS target

# Visual tuning for glass + glow
NEON_COLOR = QColor(180, 102, 255, 220)  # soft purple
GLASS_ALPHA = 0.06  # panel translucency
BUTTON_ALPHA = 0.04
BLUR_RADIUS = 12
GLOW_RADIUS = 24

# ---------------- Validate model files ----------------
if not os.path.exists("config_files"):
    raise FileNotFoundError("Missing config_files/ directory with coco.names and model files.")

names_path = "config_files/coco.names"
configPath = 'config_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'config_files/frozen_inference_graph.pb'
if not (os.path.exists(names_path) and os.path.exists(configPath) and os.path.exists(weightsPath)):
    raise FileNotFoundError("Model files missing in config_files/. Please add coco.names, .pb and .pbtxt")

with open(names_path, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# ---------------- Centroid tracker ----------------
class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=80):
        self.next_id = 0
        self.objects = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
        if oid in self.disappeared:
            del self.disappeared[oid]

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (sx, sy, ex, ey)) in enumerate(rects):
            input_centroids[i] = ((sx + ex)//2, (sy + ey)//2)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
        else:
            oids = list(self.objects.keys())
            ocentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(ocentroids)[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (r, c) in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                oid = oids[r]
                self.objects[oid] = tuple(input_centroids[c])
                self.disappeared[oid] = 0
                used_rows.add(r); used_cols.add(c)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            for r in unused_rows:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            for c in unused_cols:
                self.register(tuple(input_centroids[c]))

        return self.objects

# ---------------- Detector engine (lightweight) ----------------
class DetectorEngine:
    def __init__(self):
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self.objid_class = dict()
        self.counted_ids = set()
        self.class_counts = dict()
        self.enter_time = dict()
        self.centroid_history = defaultdict(list)
        self.abandoned_candidates = dict()
        self.heatmap = None

    def reset(self):
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self.objid_class.clear()
        self.counted_ids.clear()
        self.class_counts.clear()
        self.enter_time.clear()
        self.centroid_history.clear()
        self.abandoned_candidates.clear()
        self.heatmap = None

    def init_heatmap(self, h, w):
        self.heatmap = np.zeros((h, w), dtype=np.float32)

    def add_heat(self, cx, cy, radius=28):
        if self.heatmap is None:
            return
        rr = radius
        y1 = max(0, cy-rr); y2 = min(self.heatmap.shape[0], cy+rr)
        x1 = max(0, cx-rr); x2 = min(self.heatmap.shape[1], cx+rr)
        # simple circular stamp (fast)
        for yy in range(y1, y2):
            dy = yy - cy
            for xx in range(x1, x2):
                dx = xx - cx
                if dx*dx + dy*dy <= rr*rr:
                    self.heatmap[yy, xx] = min(self.heatmap[yy, xx] + 1.0, 255.0)

    def get_heatmap_visual(self):
        if self.heatmap is None:
            return None
        norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cmap

    def detect_post(self, frame, target_object=None):
        h, w = frame.shape[:2]
        if self.heatmap is None:
            self.init_heatmap(h, w)

        classIds, confs, bboxes = net.detect(frame, confThreshold=THRESH)
        rects = []
        detections = []
        if isinstance(bboxes, tuple):
            bboxes = list(bboxes)

        if len(bboxes) > 0 and len(confs) > 0:
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            indices = cv2.dnn.NMSBoxes(bboxes, confs, THRESH, NMS_THRESH)
            if len(indices) > 0:
                for i in indices:
                    i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
                    x, y, bw, bh = bboxes[i]
                    sx, sy, ex, ey = int(x), int(y), int(x + bw), int(y + bh)
                    cls = classNames[int(classIds[i]) - 1]
                    if target_object and cls.lower() != target_object:
                        continue
                    detections.append((sx, sy, ex, ey, cls, confs[i]))
                    rects.append((sx, sy, ex, ey))

        objects = self.tracker.update(rects)
        box_centroids = [((sx+ex)//2, (sy+ey)//2) for (sx,sy,ex,ey) in rects]

        # draw detection boxes first (fast)
        for (sx, sy, ex, ey, cls, conf) in detections:
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0,255,0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (sx+6, sy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # abandoned object candidate (light)
            if cls in OBJECT_CLASSES_TO_TRACK:
                key = f"{sx}_{sy}_{ex}_{ey}_{cls}"
                now = time.time()
                if key not in self.abandoned_candidates:
                    self.abandoned_candidates[key] = {"first": now, "last_move": now, "bbox": (sx,sy,ex,ey), "flagged": False}
                else:
                    old_bbox = self.abandoned_candidates[key]["bbox"]
                    old_c = ((old_bbox[0]+old_bbox[2])//2, (old_bbox[1]+old_bbox[3])//2)
                    new_c = ((sx+ex)//2, (sy+ey)//2)
                    if math.hypot(old_c[0]-new_c[0], old_c[1]-new_c[1]) > 10:
                        self.abandoned_candidates[key]["last_move"] = now
                        self.abandoned_candidates[key]["bbox"] = (sx,sy,ex,ey)
                    if (not self.abandoned_candidates[key]["flagged"]) and (now - self.abandoned_candidates[key]["last_move"] > ABANDONED_SECONDS):
                        # check for nearby person
                        person_near = False
                        for oid,c in self.tracker.objects.items():
                            if math.hypot(c[0]-new_c[0], c[1]-new_c[1]) < ABANDONED_DISTANCE:
                                person_near = True
                                break
                        if not person_near:
                            self.abandoned_candidates[key]["flagged"] = True

        # annotate tracked objects
        for oid, centroid in list(objects.items()):
            now = time.time()
            if oid not in self.enter_time:
                self.enter_time[oid] = now
            self.centroid_history[oid].append((centroid, now))
            assigned = self.objid_class.get(oid, None)
            if box_centroids:
                dists = [math.hypot(centroid[0]-bc[0], centroid[1]-bc[1]) for bc in box_centroids]
                idx = int(np.argmin(dists)) if len(dists) > 0 else None
                if idx is not None and idx < len(detections):
                    assigned = detections[idx][4]
                    self.objid_class[oid] = assigned

            label = f"{self.objid_class.get(oid,'Unknown')} ID{oid}"
            cv2.putText(frame, label, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,255), -1)

            # counting
            line_y = int(h * ROI_LINE_REL)
            if oid not in self.counted_ids and centroid[1] > line_y:
                cls = self.objid_class.get(oid, "Unknown")
                self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
                self.counted_ids.add(oid)

            # heat accumulation (light)
            self.add_heat(centroid[0], centroid[1], radius=24)

        # draw tripwire & counts
        line_y = int(frame.shape[0] * ROI_LINE_REL)
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0,0,255), 2)
        cv2.putText(frame, "TRIPWIRE", (10, line_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        y0 = 30
        for cls, cnt in self.class_counts.items():
            cv2.putText(frame, f"{cls}: {cnt}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            y0 += 24

        abandoned_list = [k for k,v in self.abandoned_candidates.items() if v.get("flagged", False)]
        return frame, abandoned_list

# ---------------- Motion detector ----------------
class MotionDetector:
    def __init__(self):
        self.prev_gray = None
        self.last_motion_time = time.time()

    def check(self, gray_frame):
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            self.last_motion_time = time.time()
            return True
        diff = cv2.absdiff(self.prev_gray, gray_frame)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_found = False
        for c in contours:
            if cv2.contourArea(c) > MOTION_MIN_AREA:
                motion_found = True
                break
        if motion_found:
            self.last_motion_time = time.time()
        self.prev_gray = gray_frame
        return motion_found

# ---------------- UI / Effects helpers ----------------
def apply_glow_effect(widget, radius=GLOW_RADIUS, color=NEON_COLOR, blur=18):
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setColor(color)
    effect.setOffset(0, 0)
    widget.setGraphicsEffect(effect)

def apply_blur(widget, radius=BLUR_RADIUS):
    blur = QGraphicsBlurEffect(widget)
    blur.setBlurRadius(radius)
    widget.setGraphicsEffect(blur)

# Custom Glass Button class to toggle glow on hover
class GlassButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(44)
        self.setStyleSheet(self.base_style())
        # create shadow effect but keep low intensity; hover will scale its opacity
        self.glow = QGraphicsDropShadowEffect(self)
        self.glow.setBlurRadius(0)
        self.glow.setColor(NEON_COLOR)
        self.glow.setOffset(0, 0)
        self.setGraphicsEffect(self.glow)

    def base_style(self):
        return f"""
            QPushButton {{
                background: rgba(255,255,255,{int(BUTTON_ALPHA*255)});
                color: white;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.08);
                font-weight: 700;
                padding-left: 14px;
                padding-right: 14px;
            }}
        """

    def enterEvent(self, e):
        # animate glow
        self._anim_glow(blur=22)
        self.setStyleSheet(f"""
            QPushButton {{
                background: rgba(255,255,255,{int(BUTTON_ALPHA*255 + 10)});
                color: white;
                border-radius: 12px;
                border: 1px solid rgba(180,102,255,0.28);
                font-weight: 700;
            }}
        """)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._anim_glow(blur=0)
        self.setStyleSheet(self.base_style())
        super().leaveEvent(e)

    def _anim_glow(self, blur=0):
        # simple setter (no smooth animation to keep code light)
        self.glow.setBlurRadius(blur)

# ---------------- MainWindow ----------------
class GlassMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DetectorVision — Neo Glass (Neon Purple, Strong Frosted)")
        self.setMinimumSize(1250, 820)

        # animated gradient params
        self._grad_pos = 0.0
        self._grad_dir = 1.0

        # central layout
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(20,20,20,20)
        main_h.setSpacing(18)

        # left glass sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar_glass")
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(360)
        sidebar.setStyleSheet(self.sidebar_style())
        # apply blur effect on sidebar
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(BLUR_RADIUS)
        sidebar.setGraphicsEffect(blur_effect)

        side_v = QVBoxLayout(sidebar)
        side_v.setContentsMargins(18,18,18,18)
        side_v.setSpacing(12)

        title = QLabel("DetectorVision")
        title.setStyleSheet("color: white; font-size: 22px; font-weight: 800;")
        side_v.addWidget(title)

        sub = QLabel("Neo-Glass Security Dashboard")
        sub.setStyleSheet("color: rgba(255,255,255,0.75); font-size: 11px;")
        side_v.addWidget(sub)

        side_v.addSpacing(6)

        # buttons
        self.btn_start = GlassButton("Start Live Feed")
        self.btn_start.clicked.connect(self.start_camera)
        side_v.addWidget(self.btn_start)

        self.btn_stop = GlassButton("Stop Live Feed")
        self.btn_stop.clicked.connect(self.stop_camera)
        side_v.addWidget(self.btn_stop)

        self.btn_img = GlassButton("Detect From Image")
        self.btn_img.clicked.connect(self.load_image)
        side_v.addWidget(self.btn_img)

        self.btn_vid = GlassButton("Detect From Video")
        self.btn_vid.clicked.connect(self.load_video)
        side_v.addWidget(self.btn_vid)

        side_v.addSpacing(6)

        self.lbl_target = QLabel("Target (optional):")
        self.lbl_target.setStyleSheet("color: rgba(255,255,255,0.9);")
        side_v.addWidget(self.lbl_target)
        self.target_label = QLabel("(none)")
        self.target_label.setStyleSheet("color: rgba(220,220,220,0.9); font-weight:600;")
        side_v.addWidget(self.target_label)

        side_v.addSpacing(8)

        self.btn_heat = GlassButton("Show Heatmap (on-demand)")
        self.btn_heat.clicked.connect(self.show_heatmap)
        side_v.addWidget(self.btn_heat)

        self.btn_reset = GlassButton("Reset Tracker")
        self.btn_reset.clicked.connect(self.reset_tracker)
        side_v.addWidget(self.btn_reset)

        side_v.addItem(QSpacerItem(20,20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        footer = QLabel("Neon Purple • Glass UI")
        footer.setStyleSheet("color: rgba(255,255,255,0.6); font-size: 11px;")
        side_v.addWidget(footer)

        # main glass panel
        main_panel = QFrame()
        main_panel.setObjectName("main_glass")
        main_panel.setStyleSheet(self.main_panel_style())
        mp_v = QVBoxLayout(main_panel)
        mp_v.setContentsMargins(14,14,14,14)
        mp_v.setSpacing(10)

        # video area (frosted card)
        self.video_label = QLabel()
        self.video_label.setFixedSize(FRAME_W, FRAME_H)
        self.video_label.setStyleSheet("background: rgba(0,0,0,0.32); border-radius: 14px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        mp_v.addWidget(self.video_label, alignment=Qt.AlignHCenter)

        # status & fps row
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: rgba(255,255,255,0.9); font-weight:600;")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: rgba(255,255,255,0.9); font-weight:600;")
        status_row.addWidget(self.fps_label)
        mp_v.addLayout(status_row)

        # add to main layout
        main_h.addWidget(sidebar)
        main_h.addWidget(main_panel, stretch=1)

        # detection & motion
        self.detector = DetectorEngine()
        self.motion = MotionDetector()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.detection_enabled = True
        self._last_fps_time = time.time()
        self._frame_count = 0

        # animate gradient with QTimer
        self._bg_timer = QTimer()
        self._bg_timer.timeout.connect(self._animate_background)
        self._bg_timer.start(60)  # 60 ms for smooth animation

        # initial glow on start button to show accent
        apply_glow_effect(self.btn_start, radius=GLOW_RADIUS, color=NEON_COLOR, blur=14)

    # styles
    def sidebar_style(self):
        return f"""
            QFrame#sidebar_glass {{
                background: rgba(255,255,255,{int(GLASS_ALPHA*255)});
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.06);
            }}
        """

    def main_panel_style(self):
        return f"""
            QFrame#main_glass {{
                background: rgba(255,255,255,{int(GLASS_ALPHA*255)});
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.06);
            }}
        """

    # background animation: change internal _grad_pos and repaint
    def _animate_background(self):
        # move gradient position back and forth
        step = 0.004
        self._grad_pos += step * self._grad_dir
        if self._grad_pos > 1.0:
            self._grad_pos = 1.0
            self._grad_dir = -1.0
        elif self._grad_pos < 0.0:
            self._grad_pos = 0.0
            self._grad_dir = 1.0
        self.update()  # triggers paintEvent

    def paintEvent(self, event):
        # paint animated gradient background behind widgets
        painter = QPainter(self)
        rect = self.rect()
        g = QLinearGradient(rect.topLeft(), rect.bottomRight())
        # color stops change with _grad_pos for subtle movement
        rpos = self._grad_pos
        g.setColorAt(0.0, QColor.fromRgbF(0.06 + 0.02*rpos, 0.02 + 0.01*rpos, 0.12 + 0.03*rpos, 1.0))
        g.setColorAt(0.5, QColor(24 + int(8*rpos), 6 + int(4*rpos), 50 + int(20*rpos)))
        g.setColorAt(1.0, QColor(35, 12, 60))
        brush = QBrush(g)
        painter.fillRect(rect, brush)
        super().paintEvent(event)

    # camera control
    def start_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Unable to open camera.")
            self.cap = None
            return
        self.detector.reset()
        self.motion = MotionDetector()
        self.detection_enabled = True
        self._frame_count = 0
        self._last_fps_time = time.time()
        self.timer.start(TIMER_INTERVAL_MS)
        self.status_label.setText("Live feed running. Detection active.")

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.video_label.clear()
        self.status_label.setText("Camera stopped.")
        self.fps_label.setText("FPS: 0")

    def _on_timer(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21,21), 0)

        motion = self.motion.check(gray_blur)
        now = time.time()
        if (now - self.motion.last_motion_time) > MOTION_PAUSE_SECONDS:
            if self.detection_enabled:
                self.detection_enabled = False
                self.detector.reset()
                self.status_label.setText("No motion → detection paused. Camera still on.")
        else:
            if not self.detection_enabled:
                self.detection_enabled = True
                self.detector.reset()
                self.status_label.setText("Motion detected → detection resumed.")

        display = frame.copy()
        start = time.time()
        abandoned = []
        if self.detection_enabled:
            processed, abandoned = self.detector.detect_post(frame, target_object=None)
            display = processed
        fps = 1.0 / max(1e-6, time.time() - start)
        # smooth fps display
        self._frame_count += 1
        if time.time() - self._last_fps_time >= 0.6:
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._last_fps_time = time.time()
            self._frame_count = 0

        if abandoned:
            # one-time warning (non-intrusive)
            QMessageBox.warning(self, "Abandoned Objects", f"{len(abandoned)} possible abandoned objects detected.")

        # show frame in video_label
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # file handlers
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "File Error", "Unable to open image.")
            return
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        self.detector.reset()
        processed, _ = self.detector.detect_post(img)
        cv2.imshow("Image - Detection (press any key to close)", processed)
        cv2.waitKey(0)
        cv2.destroyWindow("Image - Detection (press any key to close)")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "File Error", "Unable to open video.")
            return
        self.detector.reset()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            processed, _ = self.detector.detect_post(frame)
            cv2.imshow("Video - Detection (press q to quit)", processed)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def show_heatmap(self):
        cmap = self.detector.get_heatmap_visual()
        if cmap is None:
            QMessageBox.information(self, "Heatmap", "No activity recorded yet.")
            return
        cv2.imshow("Activity Heatmap (press any key to close)", cmap)
        cv2.waitKey(0)
        cv2.destroyWindow("Activity Heatmap (press any key to close)")

    def reset_tracker(self):
        self.detector.reset()
        QMessageBox.information(self, "Reset", "Tracker and counters reset.")

    def closeEvent(self, event):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        event.accept()

# ---------------- Run ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GlassMainWindow()
    window.show()
    sys.exit(app.exec())
