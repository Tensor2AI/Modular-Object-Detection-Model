# Modular Object Detection 

A **high-performance real-time object detection and tracking system** built with **OpenCV, SSD MobileNet v3, and PySide6**.  
This project provides a **modern GUI** for monitoring live camera feeds, detecting objects, tracking motion, and generating visual analytics such as activity heatmaps.

The application is designed to be **lightweight, responsive, and modular**, enabling easy experimentation with real-time computer vision pipelines.

---

# Key Features

- ⚡ **Real-time Object Detection**
  - SSD MobileNet v3 COCO model via OpenCV DNN
  - Fast inference with Non-Maximum Suppression (NMS)

- 🎯 **Centroid Object Tracking**
  - Persistent IDs assigned to detected objects
  - Tracks objects across frames

- 📊 **Live GUI Dashboard**
  - Built using **PySide6**
  - Status cards for:
    - Detection status
    - Object counts
    - FPS performance

- 📈 **Activity Heatmap**
  - Visualizes high-traffic movement zones

- 🚶 **Motion-Based Detection Pause**
  - Automatically pauses detection when no motion is detected

- 🔴 **Tripwire Counting System**
  - Counts objects crossing a configurable ROI line

- ⚠ **Abandoned Object Detection**
  - Flags suspicious stationary objects (bags, backpacks, etc.)

- 🖼 **Multiple Input Modes**
  - Live camera feed
  - Video file detection
  - Image detection

## Dependencies

The project requires the following Python libraries:

| Library | Purpose |
|-------|--------|
| PySide6 | GUI framework used to build the application interface |
| OpenCV (opencv-python) | Object detection, image processing, and video capture |
| NumPy | Numerical computations and array operations |
| Pillow | Image handling and conversion support |

# Installation

### 1. Clone the repository
### 2. Install dependencies
### 3. Run Main.py



## Use Cases

### Smart Surveillance Systems
Deploy as an AI-powered monitoring system for detecting suspicious objects or unusual activity in surveillance environments.

### Retail Analytics
Analyze customer movement patterns and store traffic using object tracking and activity heatmaps.

### Crowd Monitoring
Track movement patterns in crowded environments such as events, malls, or transportation hubs.

### Smart Security Infrastructure
Detect abandoned objects or suspicious behaviors in airports, train stations, or public spaces.

### AI-Enhanced CCTV Systems
Upgrade traditional CCTV setups with **real-time object detection and analytics capabilities**.

### Footfall Counting
Use tripwire detection to measure people flow in entrances, exits, or specific zones.

### Computer Vision Research
Serve as a **framework for experimentation with detection models, tracking algorithms, and real-time AI pipelines**.

---
