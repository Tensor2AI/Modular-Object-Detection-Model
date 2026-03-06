# DetectorVision_Cyberpunk_PySide6.py
# Cyberpunk-themed PySide6 GUI integrated with fast detection + tracking.
# Place next to config_files/ (coco.names, frozen_inference_graph.pb, .pbtxt).
# No file writes, high-FPS, heatmap on-demand, motion-based pause.
#
# Requirements:
# pip install PySide6 opencv-python numpy Pillow

import sys
import time
import math
import threading
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

# -------- CONFIG --------
THRESH = 0.6
NMS_THRESH = 0.4
CAM_RES_W = 640
CAM_RES_H = 480

MOTION_MIN_AREA = 800
MOTION_PAUSE_SECONDS = 10.0

ROI_LINE_REL = 0.66
ABANDONED_SECONDS = 8
ABANDONED_DISTANCE = 80
OBJECT_CLASSES_TO_TRACK = {"backpack", "handbag", "suitcase", "bottle", "bag"}

HEATMAP_RADIUS = 30

# -------- load detection model (ssd mobilenet v3) --------
classNames = []
with open('config_files/coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'config_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'config_files/frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# -------- Simple centroid tracker --------
class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=70):
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

# -------- Application (PySide6) --------
class CyberWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DetectorVision — Cyberpunk")
        self.setMinimumSize(1200, 780)
        self.setStyleSheet(self._main_stylesheet())

        # central widget
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QHBoxLayout(central)
        central_layout.setContentsMargins(12,12,12,12)
        central_layout.setSpacing(12)
        self.setCentralWidget(central)

        # left sidebar (controls)
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setFixedWidth(260)
        self.sidebar.setObjectName("sidebar")
        side_layout = QtWidgets.QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(16,16,16,16)
        side_layout.setSpacing(10)

        title = QtWidgets.QLabel("DETECTOR\nVISION")
        title.setObjectName("appTitle")
        title.setAlignment(QtCore.Qt.AlignLeft)
        side_layout.addWidget(title)

        side_layout.addSpacing(6)

        # control buttons (neon)
        self.btn_start = self._neon_button("Start Live Feed", "#00fff5", "#8b00ff")
        self.btn_stop = self._neon_button("Stop Live Feed", "#ff4dff", "#ff8a00")
        self.btn_video = self._neon_button("Detect From Video", "#00c3ff", "#7b2cff")
        self.btn_image = self._neon_button("Detect From Image", "#00ff7a", "#00b3ff")

        side_layout.addWidget(self.btn_start)
        side_layout.addWidget(self.btn_stop)
        side_layout.addWidget(self.btn_video)
        side_layout.addWidget(self.btn_image)

        side_layout.addSpacing(8)

        self.target_label = QtWidgets.QLabel("Target Object (optional)")
        self.target_label.setObjectName("sideLabel")
        side_layout.addWidget(self.target_label)
        self.entry_target = QtWidgets.QLineEdit()
        self.entry_target.setObjectName("neonEntry")
        side_layout.addWidget(self.entry_target)
        self.btn_reset_target = self._neon_button("Reset Target", "#888", "#444", small=True)
        side_layout.addWidget(self.btn_reset_target)

        side_layout.addSpacing(6)

        # feature toggles
        self.btn_heatmap = self._neon_button("Show Heatmap (on-demand)", "#ffaa00", "#ff55aa")
        self.btn_toggle_roi = self._neon_button("Toggle ROI", "#00a3ff", "#00ffd1", small=True)
        side_layout.addWidget(self.btn_heatmap)
        side_layout.addWidget(self.btn_toggle_roi)

        side_layout.addStretch()

        self.btn_close = self._neon_button("Exit", "#ff0055", "#ff9900")
        side_layout.addWidget(self.btn_close)

        # right content: video + cards
        self.content = QtWidgets.QFrame()
        content_layout = QtWidgets.QVBoxLayout(self.content)
        content_layout.setContentsMargins(10,10,10,10)
        content_layout.setSpacing(10)

        # video area (label)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setObjectName("videoArea")
        self.video_label.setFixedSize(CAM_RES_W, CAM_RES_H)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        content_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignCenter)

        # overlay cards row
        cards_row = QtWidgets.QHBoxLayout()
        cards_row.setSpacing(12)

        # status card
        self.card_status = self._info_card("STATUS", "Ready")
        self.card_counts = self._info_card("COUNTS", "No detections")
        self.card_fps = self._info_card("FPS", "0")
        cards_row.addWidget(self.card_status)
        cards_row.addWidget(self.card_counts)
        cards_row.addWidget(self.card_fps)

        content_layout.addLayout(cards_row)

        # right-most debug panel area (small)
        self.panel_right = QtWidgets.QFrame()
        self.panel_right.setFixedWidth(240)
        panel_layout = QtWidgets.QVBoxLayout(self.panel_right)
        panel_layout.setContentsMargins(10,10,10,10)
        panel_layout.setSpacing(8)

        lbl = QtWidgets.QLabel("Quick Actions")
        lbl.setObjectName("sideLabel")
        panel_layout.addWidget(lbl)

        self.btn_reset = self._neon_button("Reset Tracker", "#8888ff", "#ff88ff", small=True)
        self.btn_show_counts = self._neon_button("Reset Counts", "#55ff88", "#66ccff", small=True)
        panel_layout.addWidget(self.btn_reset)
        panel_layout.addWidget(self.btn_show_counts)

        panel_layout.addStretch()

        # assemble
        central_layout.addWidget(self.sidebar)
        central_layout.addWidget(self.content, stretch=1)
        central_layout.addWidget(self.panel_right)

        # connect signals
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_video.clicked.connect(self.detect_from_file)
        self.btn_image.clicked.connect(self.detect_from_image)
        self.btn_close.clicked.connect(self.close)
        self.btn_reset_target.clicked.connect(self.reset_target)
        self.btn_reset.clicked.connect(self._reset_tracker)
        self.btn_show_counts.clicked.connect(self._reset_counts)
        self.btn_heatmap.clicked.connect(self._show_heatmap)
        self.btn_toggle_roi.clicked.connect(self._toggle_roi)

        # detection state variables
        self.cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._grab_frame)
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.detection_enabled = True
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self.objid_class = dict()
        self.counted_ids = set()
        self.class_counts = dict()
        self.enter_time = dict()
        self.centroid_history = defaultdict(list)
        self.abandoned_candidates = dict()
        self.heatmap_accum = None
        self.enable_roi = True

        # show window
        self.show()

    # ----- UI helpers -----
    def _neon_button(self, text, color_a="#00fff5", color_b="#8b00ff", small=False):
        btn = QtWidgets.QPushButton(text)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setStyleSheet(self._neon_btn_style(color_a, color_b, small))
        btn.setFixedHeight(40 if not small else 34)
        # drop shadow (glow)
        effect = QtWidgets.QGraphicsDropShadowEffect()
        effect.setBlurRadius(18)
        effect.setOffset(0)
        # use color_a as glow
        try:
            col = QtGui.QColor(color_a)
            effect.setColor(col)
        except Exception:
            effect.setColor(QtGui.QColor("#00ffcc"))
        btn.setGraphicsEffect(effect)
        return btn

    def _info_card(self, title, value):
        card = QtWidgets.QFrame()
        card.setObjectName("infoCard")
        layout = QtWidgets.QVBoxLayout(card)
        t = QtWidgets.QLabel(title)
        t.setObjectName("cardTitle")
        v = QtWidgets.QLabel(value)
        v.setObjectName("cardValue")
        v.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(t)
        layout.addWidget(v)
        card.value_label = v
        return card

    # ----- styles -----
    def _main_stylesheet(self):
        return """
        QMainWindow { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0c29, stop:0.5 #302b63, stop:1 #24243e); }
        QFrame#sidebar { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0b1226, stop:1 #15102b);
                         border-radius:12px; }
        QLabel#appTitle { color: #00fff5; font-size:24px; font-weight:800; }
        QLabel#sideLabel { color:#cfcfff; font-size:12px; margin-top:6px; }
        QLabel#videoArea { background: rgba(0,0,0,0.18); border-radius:8px; border: 1px solid rgba(255,255,255,0.05);}
        QFrame#infoCard { background: rgba(255,255,255,0.03); border-radius:10px; padding:10px; }
        QLabel#cardTitle { color:#bdb6ff; font-size:11px; }
        QLabel#cardValue { color:#ffffff; font-size:18px; font-weight:700; }
        QLineEdit#neonEntry { background: rgba(255,255,255,0.03); border-radius:8px; color:#fff; padding:6px; }
        QPushButton { border:none; color:#111; font-weight:700; }
        """

    def _neon_btn_style(self, a, b, small=False):
        # gradient + subtle text color
        return f"""
        QPushButton {{
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {a}, stop:1 {b});
            border-radius: 10px;
            color: white;
            padding-left:12px;
            padding-right:12px;
            font-weight:700;
        }}
        QPushButton:hover {{
            transform: scale(1.02);
        }}
        """

    # ----- detection control methods -----
    def start_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Camera error", "Unable to access the camera.")
            self.cap = None
            return
        # set resolution
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RES_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES_H)
        except Exception:
            pass
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.detection_enabled = True
        self.reset_tracker()
        self.timer.start(30)  # ~33 fps timer tick, processing kept lightweight

    def stop_camera(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.timer.stop()
        self.video_label.clear()
        self.card_status.value_label.setText("Stopped")
        self.card_fps.value_label.setText("0")

    def detect_from_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video file", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        # play with blocking cv2 to keep main UI responsive (simple)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "File error", "Cannot open selected video.")
            return
        self.reset_tracker()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (CAM_RES_W, CAM_RES_H))
            proc = self._process_frame(frame)
            cv2.imshow("Video - Detection (press q to quit)", proc)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_from_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image file", "", "Image Files (*.jpg *.png *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "File error", "Cannot open selected image.")
            return
        img = cv2.resize(img, (CAM_RES_W, CAM_RES_H))
        proc = self._process_frame(img)
        cv2.imshow("Image - Detection (press any key to close)", proc)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def reset_target(self):
        self.entry_target.clear()
        self.card_status.value_label.setText("Target cleared")

    def _reset_tracker(self):
        self.reset_tracker()
        self.card_status.value_label.setText("Tracker reset")

    def _reset_counts(self):
        self.class_counts.clear()
        self.counted_ids.clear()
        self.card_counts.value_label.setText("No detections")

    def _toggle_roi(self):
        self.enable_roi = not self.enable_roi
        self.card_status.value_label.setText("ROI ON" if self.enable_roi else "ROI OFF")

    def _show_heatmap(self):
        if self.heatmap_accum is None:
            QtWidgets.QMessageBox.information(self, "Heatmap", "No activity recorded yet.")
            return
        # normalize and show in opencv window
        norm = cv2.normalize(self.heatmap_accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imshow("Activity Heatmap (press any key)", cmap)
        cv2.waitKey(0)
        cv2.destroyWindow("Activity Heatmap (press any key)")

    # ----- processing loop -----
    def _grab_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (CAM_RES_W, CAM_RES_H))
        # motion detection (fast)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        motion = self._detect_motion(gray_blur)
        now = time.time()
        if (now - self.last_motion_time) > MOTION_PAUSE_SECONDS:
            if self.detection_enabled:
                self.detection_enabled = False
                self.card_status.value_label.setText("Paused (no motion)")
                self.reset_tracker()
        else:
            if not self.detection_enabled:
                self.detection_enabled = True
                self.card_status.value_label.setText("Resumed (motion)")
                self.reset_tracker()

        if self.detection_enabled:
            start = time.time()
            proc = self._process_frame(frame)
            fps = 1.0 / max(1e-6, time.time() - start)
            self.card_fps.value_label.setText(f"{fps:.1f}")
            self._show_in_label(proc)
        else:
            # show frame with overlay text
            disp = frame.copy()
            cv2.putText(disp, "Detection Paused - Monitoring Motion...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            self._show_in_label(disp)

    def _show_in_label(self, bgr_frame):
        # convert to QImage and set
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def _detect_and_nms(self, frame):
        classIds, confs, bboxes = net.detect(frame, confThreshold=THRESH)
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
                    detections.append((sx, sy, ex, ey, cls, confs[i]))
        return detections

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        if self.heatmap_accum is None:
            self.heatmap_accum = np.zeros((h, w), dtype=np.float32)
        detections = self._detect_and_nms(frame)
        rects = [(sx, sy, ex, ey) for (sx, sy, ex, ey, _, _) in detections]

        objects = self.tracker.update(rects)
        box_centroids = [((sx+ex)//2, (sy+ey)//2) for (sx,sy,ex,ey) in rects]

        # map classes
        for oid, centroid in list(objects.items()):
            now = time.time()
            if oid not in self.enter_time:
                self.enter_time[oid] = now
            self.centroid_history[oid].append((centroid, now))
            # assign class if possible
            assigned = self.objid_class.get(oid, None)
            if box_centroids:
                dists = [math.hypot(centroid[0]-bc[0], centroid[1]-bc[1]) for bc in box_centroids]
                idx = int(np.argmin(dists))
                if idx < len(detections):
                    assigned = detections[idx][4]
                    self.objid_class[oid] = assigned

            # draw id + class
            lab = f"{self.objid_class.get(oid,'Unknown')} ID{oid}"
            cv2.putText(frame, lab, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,255), -1)

            # counting: tripwire
            line_y = int(h * ROI_LINE_REL)
            if oid not in self.counted_ids and centroid[1] > line_y:
                cls = self.objid_class.get(oid, "Unknown")
                self.class_counts[cls] = self.class_counts.get(cls, 0) + 1
                self.counted_ids.add(oid)
                self.card_counts.value_label.setText(", ".join([f"{k}:{v}" for k,v in self.class_counts.items()]))

            # heatmap add
            cx, cy = centroid
            rr = HEATMAP_RADIUS
            y1 = max(0, cy-rr); y2 = min(self.heatmap_accum.shape[0], cy+rr)
            x1 = max(0, cx-rr); x2 = min(self.heatmap_accum.shape[1], cx+rr)
            for yy in range(y1, y2):
                dy = yy - cy
                for xx in range(x1, x2):
                    dx = xx - cx
                    if dx*dx + dy*dy <= rr*rr:
                        self.heatmap_accum[yy, xx] = min(self.heatmap_accum[yy, xx] + 1.0, 255.0)

        # draw boxes
        for (sx, sy, ex, ey, cls, conf) in detections:
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (sx+6, sy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # abandoned candidate lightweight
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
                        # check for person nearby
                        center_new = new_c
                        person_near = False
                        for oid, cent in self.tracker.objects.items():
                            if math.hypot(cent[0]-center_new[0], cent[1]-center_new[1]) < ABANDONED_DISTANCE:
                                person_near = True
                                break
                        if not person_near:
                            self.abandoned_candidates[key]["flagged"] = True
                            # show a non-blocking alert
                            QtWidgets.QMessageBox.warning(self, "Abandoned Object", f"Possible abandoned object: {cls}")

        # ROI line
        if self.enable_roi:
            line_y = int(frame.shape[0] * ROI_LINE_REL)
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
            cv2.putText(frame, "TRIPWIRE", (10, line_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # update counts card
        if self.class_counts:
            self.card_counts.value_label.setText(", ".join([f"{k}:{v}" for k,v in self.class_counts.items()]))
        else:
            self.card_counts.value_label.setText("No detections")

        return frame

    # ---------- motion detection ----------
    def _detect_motion(self, gray_blur):
        if self.prev_gray is None:
            self.prev_gray = gray_blur
            self.last_motion_time = time.time()
            return True
        diff = cv2.absdiff(self.prev_gray, gray_blur)
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
        self.prev_gray = gray_blur
        return motion_found

    def reset_tracker(self):
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self.objid_class.clear()
        self.counted_ids.clear()
        self.class_counts.clear()
        self.enter_time.clear()
        self.centroid_history.clear()
        self.abandoned_candidates.clear()
        self.heatmap_accum = None

# -------- main --------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CyberWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
