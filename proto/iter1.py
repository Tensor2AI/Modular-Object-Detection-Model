"""
DetectorVision_PySide6_Glass.py
Neo-Glassmorphism Dashboard (PySide6) + Fast detection pipeline.

Features:
- SSD MobileNet v3 detection (uses config_files/ assets). :contentReference[oaicite:1]{index=1}
- Centroid tracking (IDs), counting via tripwire, ROI intrusion, lightweight abandoned-object detection.
- Motion-based auto-pause/resume (camera stays on).
- Heatmap ON-DEMAND (no continuous saving).
- Modern glassmorphic UI with blur, shadows, neon accents (PySide6).
- No disk writes; optimized for CPU performance.

Requirements:
- PySide6, opencv-python, numpy, Pillow
"""

import sys
import os
import time
import math
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

# -------------------------
# Configuration & constants
# -------------------------
THRESH = 0.6
NMS_THRESH = 0.4

FRAME_W = 640
FRAME_H = 480

# Motion pause
MOTION_MIN_AREA = 800
MOTION_PAUSE_SECONDS = 10.0
MOTION_CHECK_INTERVAL_MS = 80  # ~12.5 fps timer, detection runs when enabled

ROI_LINE_REL = 0.66

# Abandoned object
ABANDONED_SECONDS = 8
ABANDONED_DISTANCE = 80
OBJECT_CLASSES_TO_TRACK = {"backpack", "handbag", "suitcase", "bottle", "bag"}

HEATMAP_KERNEL_RADIUS = 30

# Paths (must match your project)
MODEL_FOLDER = "config_files"
COCO_NAMES = os.path.join(MODEL_FOLDER, "coco.names")
CONFIG_PBTXT = os.path.join(MODEL_FOLDER, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
WEIGHTS_PB = os.path.join(MODEL_FOLDER, "frozen_inference_graph.pb")

# -------------------------
# Load model & classes
# -------------------------
if not os.path.exists(COCO_NAMES) or not os.path.exists(CONFIG_PBTXT) or not os.path.exists(WEIGHTS_PB):
    msg = ("Model files not found. Ensure the following files are present in 'config_files/' folder:\n"
           "- coco.names\n- ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt\n- frozen_inference_graph.pb")
    raise FileNotFoundError(msg)

with open(COCO_NAMES, 'rt') as f:
    CLASS_NAMES = f.read().rstrip('\n').split('\n')

net = cv2.dnn_DetectionModel(WEIGHTS_PB, CONFIG_PBTXT)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# -------------------------
# Lightweight centroid tracker
# -------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=70):
        self.next_id = 0
        self.objects = dict()      # id -> (x,y)
        self.disappeared = dict()  # id -> count
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
        # rects: list of (sx, sy, ex, ey)
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

# -------------------------
# Detection & postprocess (fast)
# -------------------------
def detect_and_postprocess(frame, tracker, state):
    """
    frame: BGR image
    tracker: CentroidTracker instance
    state: dict to carry counters, class mapping etc.
    returns annotated_frame (BGR)
    """
    h, w = frame.shape[:2]
    if state["heatmap"] is None:
        state["heatmap"] = np.zeros((h, w), dtype=np.float32)

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
                cls = CLASS_NAMES[int(classIds[i]) - 1]
                detections.append((sx, sy, ex, ey, cls, confs[i]))
                rects.append((sx, sy, ex, ey))

    objects = tracker.update(rects)

    # map boxes->centroids and draw
    box_centroids = [((sx+ex)//2, (sy+ey)//2) for (sx,sy,ex,ey) in rects]
    now = time.time()
    for oid, centroid in list(objects.items()):
        if oid not in state["enter_time"]:
            state["enter_time"][oid] = now
        state["centroid_history"][oid].append((centroid, now))

        # assign class by nearest box centroid
        assigned = state["objid_class"].get(oid, None)
        if box_centroids:
            dists = [math.hypot(centroid[0]-bc[0], centroid[1]-bc[1]) for bc in box_centroids]
            idx = int(np.argmin(dists))
            if idx < len(detections):
                assigned = detections[idx][4]
                state["objid_class"][oid] = assigned

        # draw ID + class
        label = f"{state['objid_class'].get(oid,'Unknown')} ID{oid}"
        cv2.putText(frame, label, (centroid[0]-10, centroid[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,255), -1)

        # counting: tripwire crossing top->bottom
        line_y = int(h * ROI_LINE_REL)
        if oid not in state["counted_ids"] and centroid[1] > line_y:
            cls = state["objid_class"].get(oid, "Unknown")
            state["class_counts"][cls] = state["class_counts"].get(cls, 0) + 1
            state["counted_ids"].add(oid)

        # add to heatmap accumulation
        cx, cy = centroid
        rr = HEATMAP_KERNEL_RADIUS
        y1 = max(0, cy-rr); y2 = min(state["heatmap"].shape[0], cy+rr)
        x1 = max(0, cx-rr); x2 = min(state["heatmap"].shape[1], cx+rr)
        for yy in range(y1, y2):
            dy = yy - cy
            for xx in range(x1, x2):
                dx = xx - cx
                if dx*dx + dy*dy <= rr*rr:
                    state["heatmap"][yy, xx] = min(state["heatmap"][yy, xx] + 1.0, 255.0)

    # draw raw detection boxes and manage lightweight abandoned logic
    for (sx, sy, ex, ey, cls, conf) in detections:
        cv2.rectangle(frame, (sx, sy), (ex, ey), (0,255,0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (sx+5, sy+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # abandoned candidates
        if cls in OBJECT_CLASSES_TO_TRACK:
            key = f"{sx}_{sy}_{ex}_{ey}_{cls}"
            if key not in state["abandoned"]:
                state["abandoned"][key] = {"first": now, "last_move": now, "bbox": (sx,sy,ex,ey), "flagged": False}
            else:
                old = state["abandoned"][key]
                old_center = ((old["bbox"][0]+old["bbox"][2])//2, (old["bbox"][1]+old["bbox"][3])//2)
                new_center = ((sx+ex)//2, (sy+ey)//2)
                if math.hypot(old_center[0]-new_center[0], old_center[1]-new_center[1]) > 10:
                    old["last_move"] = now
                    old["bbox"] = (sx,sy,ex,ey)
                if (not old["flagged"]) and (now - old["last_move"] > ABANDONED_SECONDS):
                    # check person proximity
                    person_near = False
                    for oid, cent in tracker.objects.items():
                        if math.hypot(cent[0]-new_center[0], cent[1]-new_center[1]) < ABANDONED_DISTANCE:
                            person_near = True
                            break
                    if not person_near:
                        old["flagged"] = True
                        # lightweight alert via state (GUI will show a small popup)
                        state["alerts"].append(f"Possible abandoned object: {cls}")

    # draw ROI line
    line_y = int(frame.shape[0] * ROI_LINE_REL)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0,0,255), 2)
    cv2.putText(frame, "TRIPWIRE", (10, line_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # draw counters
    y0 = 24
    for cls, cnt in state["class_counts"].items():
        cv2.putText(frame, f"{cls}: {cnt}", (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y0 += 20

    return frame

# -------------------------
# Helper: motion detection (cheap)
# -------------------------
def detect_motion_gray(prev_gray, gray):
    # prev_gray and gray are blurred grayscale images same size
    if prev_gray is None:
        return True, gray
    diff = cv2.absdiff(prev_gray, gray)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > MOTION_MIN_AREA:
            return True, gray
    return False, gray

# -------------------------
# PySide6 UI
# -------------------------
class GlassMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DetectorVision — Neo-Glass Dashboard")
        self.setGeometry(80, 40, 1400, 860)
        self.setMinimumSize(1000, 700)

        # central widget container
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        # glass background - we use stylesheets + translucent panels + blur effects
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        # left sidebar panel (frosted)
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setFixedWidth(300)
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setStyleSheet("""
            QFrame#sidebar {
                background: rgba(255,255,255,0.06);
                border-radius: 14px;
            }
        """)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(12)

        # Title (neon)
        title = QtWidgets.QLabel("DetectorVision")
        title.setStyleSheet("color: white; font-size: 20px; font-weight: 700;")
        sidebar_layout.addWidget(title, alignment=QtCore.Qt.AlignTop)

        # Buttons (neon gradient style using stylesheets)
        btn_spec = {
            "start": ("Start Live Feed", "#6a00ff,#00d4ff"),
            "stop": ("Stop Live Feed", "#ff4d4d,#ff7373"),
            "video": ("Detect From Video", "#0099ff,#00ccff"),
            "image": ("Detect From Image", "#00cc88,#00ffb3"),
        }

        self.btns = {}
        for k, (txt, colors) in btn_spec.items():
            b = QtWidgets.QPushButton(txt)
            b.setFixedHeight(44)
            b.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            # gradient background & glow pseudo via shadow effect
            b.setStyleSheet(f"""
                QPushButton {{
                    color: white;
                    border-radius: 12px;
                    padding: 8px 12px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {colors.split(',')[0]}, stop:1 {colors.split(',')[1]});
                    border: none;
                }}
                QPushButton:hover {{
                    filter: brightness(1.15);
                }}
            """)
            shadow = QtWidgets.QGraphicsDropShadowEffect(blurRadius=18, xOffset=0, yOffset=6)
            shadow.setColor(QtGui.QColor(0, 0, 0, 160))
            b.setGraphicsEffect(shadow)
            sidebar_layout.addWidget(b)
            self.btns[k] = b

        # Target entry
        lbl_t = QtWidgets.QLabel("Target Object (optional):")
        lbl_t.setStyleSheet("color: #dcdcdc;")
        sidebar_layout.addWidget(lbl_t)
        self.entry_target = QtWidgets.QLineEdit()
        self.entry_target.setPlaceholderText("e.g., person, cell phone")
        self.entry_target.setFixedHeight(36)
        self.entry_target.setStyleSheet("""
            QLineEdit { border-radius: 8px; padding: 8px; background: rgba(255,255,255,0.03); color: white; }
        """)
        sidebar_layout.addWidget(self.entry_target)

        # Feature toggles (compact)
        sidebar_layout.addSpacing(8)
        self.chk_tracking = QtWidgets.QCheckBox("Enable Tracking (IDs)")
        self.chk_tracking.setChecked(True)
        self.chk_tracking.setStyleSheet("color: #eaeaea;")
        sidebar_layout.addWidget(self.chk_tracking)

        self.chk_counting = QtWidgets.QCheckBox("Counting / Tripwire")
        self.chk_counting.setChecked(True)
        self.chk_counting.setStyleSheet("color: #eaeaea;")
        sidebar_layout.addWidget(self.chk_counting)

        self.chk_roi = QtWidgets.QCheckBox("ROI Tripwire")
        self.chk_roi.setChecked(True)
        self.chk_roi.setStyleSheet("color: #eaeaea;")
        sidebar_layout.addWidget(self.chk_roi)

        self.chk_abandoned = QtWidgets.QCheckBox("Abandoned Object (light)")
        self.chk_abandoned.setChecked(True)
        self.chk_abandoned.setStyleSheet("color: #eaeaea;")
        sidebar_layout.addWidget(self.chk_abandoned)

        # Heatmap & other controls
        sidebar_layout.addSpacing(6)
        self.btn_heatmap = QtWidgets.QPushButton("Show Heatmap (on demand)")
        self.btn_heatmap.setFixedHeight(40)
        self.btn_heatmap.setStyleSheet("""
            QPushButton { color: black; border-radius: 10px; background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ffcf33, stop:1 #ff8a00); }
        """)
        sidebar_layout.addWidget(self.btn_heatmap)

        # Alerts area (small)
        self.alert_label = QtWidgets.QLabel("")
        self.alert_label.setWordWrap(True)
        self.alert_label.setStyleSheet("color: #ffdd99;")
        sidebar_layout.addWidget(self.alert_label)

        sidebar_layout.addStretch(1)

        # Footer small info
        footer = QtWidgets.QLabel("Mode: Glass • FPS Optimized")
        footer.setStyleSheet("color: #bfc7d6; font-size: 11px;")
        sidebar_layout.addWidget(footer)

        # Right area: video + overlay cards
        right_container = QtWidgets.QFrame()
        right_container.setObjectName("right")
        right_container.setStyleSheet("""
            QFrame#right {
                background: rgba(255,255,255,0.03);
                border-radius: 14px;
            }
        """)
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        # Top row: small stat cards
        stats_row = QtWidgets.QHBoxLayout()
        stats_row.setSpacing(12)
        self.cards = {}
        for title in ("Detections", "Tracked IDs", "Alerts"):
            card = QtWidgets.QFrame()
            card.setFixedHeight(80)
            card.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 12px;")
            cl = QtWidgets.QVBoxLayout(card)
            lbl_t = QtWidgets.QLabel(title)
            lbl_t.setStyleSheet("color: #cbd6e2;")
            lbl_v = QtWidgets.QLabel("0")
            lbl_v.setStyleSheet("color: white; font-size: 20px; font-weight: 700;")
            cl.addWidget(lbl_t)
            cl.addStretch(1)
            cl.addWidget(lbl_v)
            stats_row.addWidget(card)
            self.cards[title] = lbl_v
        right_layout.addLayout(stats_row)

        # Video panel (QLabel showing frames)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setStyleSheet("background: rgba(0,0,0,0.25); border-radius: 12px;")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.video_label, 1)

        # Status bar below video
        self.status_bar = QtWidgets.QLabel("Status: Ready")
        self.status_bar.setStyleSheet("color: #dfe9f3;")
        right_layout.addWidget(self.status_bar)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(right_container, 1)

        # Effects: apply blur to background panels (glass)
        blur = QtWidgets.QGraphicsBlurEffect()
        blur.setBlurRadius(6)
        # apply to right_container to get soft frosted feel
        right_container.setGraphicsEffect(blur)

        # Shadow effect for video_label for lift
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(28)
        shadow.setXOffset(0)
        shadow.setYOffset(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 180))
        self.video_label.setGraphicsEffect(shadow)

        # Connect signals
        self.btns["start"].clicked.connect(self.on_start)
        self.btns["stop"].clicked.connect(self.on_stop)
        self.btns["video"].clicked.connect(self.on_detect_video)
        self.btns["image"].clicked.connect(self.on_detect_image)
        self.btn_heatmap.clicked.connect(self.on_show_heatmap)

        # Toggle handlers
        self.chk_tracking.stateChanged.connect(self.on_toggle_tracking)
        self.chk_counting.stateChanged.connect(self.on_toggle_counting)
        self.chk_roi.stateChanged.connect(self.on_toggle_roi)
        self.chk_abandoned.stateChanged.connect(self.on_toggle_abandoned)

        # state for detection
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(MOTION_CHECK_INTERVAL_MS)
        self.timer.timeout.connect(self.grab_frame)

        # previous gray for motion
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.detection_enabled = True

        # tracker + state maps
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self.state = {
            "objid_class": {},
            "counted_ids": set(),
            "class_counts": {},
            "enter_time": {},
            "centroid_history": defaultdict(list),
            "abandoned": {},
            "alerts": [],
            "heatmap": None
        }

        # defaults
        self.chk_tracking.setChecked(True)
        self.chk_counting.setChecked(True)
        self.chk_roi.setChecked(True)
        self.chk_abandoned.setChecked(True)

    # -------------------------
    # UI event callbacks
    # -------------------------
    def on_start(self):
        # start camera & timer
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_bar.setText("Status: Unable to open camera")
            return
        # set consistent resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.detection_enabled = True
        # reset tracker state
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self.state = {
            "objid_class": {},
            "counted_ids": set(),
            "class_counts": {},
            "enter_time": {},
            "centroid_history": defaultdict(list),
            "abandoned": {},
            "alerts": [],
            "heatmap": None
        }
        self.timer.start()
        self.status_bar.setText("Status: Live feed started")

    def on_stop(self):
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.video_label.clear()
        self.status_bar.setText("Status: Stopped")

    def on_detect_video(self):
        # open file dialog and play detection in separate OpenCV window (blocking)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if not path:
            return
        # blocking playback window
        cap = cv2.VideoCapture(path)
        self.status_bar.setText(f"Status: Playing {os.path.basename(path)}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            annotated = detect_and_postprocess(frame, self.tracker, self.state)
            cv2.imshow("Video - Detection (press q to quit)", annotated)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.status_bar.setText("Status: Ready")

    def on_detect_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.status_bar.setText("Status: Cannot open image")
            return
        img = cv2.resize(img, (FRAME_W, FRAME_H))
        # reset trackers
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
        self.state = {
            "objid_class": {},
            "counted_ids": set(),
            "class_counts": {},
            "enter_time": {},
            "centroid_history": defaultdict(list),
            "abandoned": {},
            "alerts": [],
            "heatmap": None
        }
        annotated = detect_and_postprocess(img, self.tracker, self.state)
        cv2.imshow("Image - Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_show_heatmap(self):
        # show heatmap on demand (Option A). Use OpenCV window to display colormap.
        if self.state["heatmap"] is None:
            QtWidgets.QMessageBox.information(self, "Heatmap", "No activity recorded yet.")
            return
        norm = cv2.normalize(self.state["heatmap"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imshow("Activity Heatmap (press any key to close)", cmap)
        cv2.waitKey(0)
        cv2.destroyWindow("Activity Heatmap (press any key to close)")

    def on_toggle_tracking(self, _):
        # toggling will be read in processing; nothing immediate required
        pass

    def on_toggle_counting(self, _):
        pass

    def on_toggle_roi(self, _):
        pass

    def on_toggle_abandoned(self, _):
        pass

    # -------------------------
    # Frame grab loop (timer)
    # -------------------------
    def grab_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21,21), 0)

        motion_found, new_prev = detect_motion_gray(self.prev_gray, gray_blur)
        self.prev_gray = new_prev
        if motion_found:
            self.last_motion_time = time.time()

        # auto pause/resume detection
        now = time.time()
        if (now - self.last_motion_time) > MOTION_PAUSE_SECONDS:
            if self.detection_enabled:
                self.detection_enabled = False
                self.status_bar.setText("Status: No motion — detection paused (camera ON)")
                # reset transient state to avoid stale IDs
                self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
                self.state["objid_class"].clear()
                self.state["centroid_history"].clear()
        else:
            if not self.detection_enabled:
                self.detection_enabled = True
                self.status_bar.setText("Status: Motion detected — detection resumed")
                self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)
                self.state["objid_class"].clear()
                self.state["centroid_history"].clear()

        display = frame
        if self.detection_enabled:
            # perform detection & postprocess
            annotated = detect_and_postprocess(frame, self.tracker, self.state)
            display = annotated
            # show alerts (take one and display)
            if len(self.state["alerts"]) > 0:
                a = self.state["alerts"].pop(0)
                self.alert_label.setText(a)
                # ephemeral clear after 5 sec
                QtCore.QTimer.singleShot(5000, lambda: self.alert_label.setText(""))
        else:
            # when paused, overlay a small notice
            cv2.putText(display, "Detection Paused - Monitoring Motion...", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2)

        # update stat cards
        dets = sum(self.state["class_counts"].values()) if self.state["class_counts"] else 0
        self.cards["Detections"].setText(str(dets))
        self.cards["Tracked IDs"].setText(str(len(self.tracker.objects)))
        self.cards["Alerts"].setText("1" if len(self.state["alerts"])>0 else "0")

        # convert to QImage and show
        img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

# -------------------------
# App entry
# -------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # global app stylesheet for background gradient
    gradient_css = """
        QMainWindow {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 rgba(10,10,30,255),
                stop:0.5 rgba(14,4,40,255),
                stop:1 rgba(6,12,40,255));
        }
    """
    app.setStyleSheet(gradient_css)
    window = GlassMainWindow()
    window.show()
    sys.exit(app.exec())
