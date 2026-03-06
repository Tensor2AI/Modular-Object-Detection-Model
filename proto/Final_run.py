import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import gradio as gr


#DETECTION PARAMETER
target_object = None
thres = 0.6
nms_threshold = 0.4
stop_flag = False
live_cap = None
current_theme = "dark"

#LOAD MODELS AND FILES
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

#MAIN DETECTION LOGIC
def detect_objects(frame, log_list=None):
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    names_detected = []

    if len(indices) > 0:
        for i in indices:
            i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            class_name = classNames[int(classIds[i]) - 1]

            if target_object and class_name.lower() != target_object:
                continue

            names_detected.append(class_name)

            if log_list is not None:
                log_list.append(class_name)

            x, y, w, h = bbox[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame, names_detected

#LIVE DETECTION FUNCTION
def show_frame():
    if stop_flag:
        return
    success, img = live_cap.read()
    if success:
        img, _ = detect_objects(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, show_frame)

def start_live_cam():
    global live_cap, stop_flag
    stop_flag = False
    live_cap = cv2.VideoCapture(0)
    show_frame()

def stop_live_cam():
    global stop_flag, live_cap
    stop_flag = True
    if live_cap:
        live_cap.release()

#FILE BASED DETECTION
def detect_video():
    path = filedialog.askopenfilename()
    if not path:
        return
    cap = cv2.VideoCapture(path)
    log_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = detect_objects(frame, log_list)
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    update_log(log_list)

def detect_image():
    path = filedialog.askopenfilename()
    if not path:
        return
    img = cv2.imread(path)
    log_list = []
    img, _ = detect_objects(img, log_list)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    update_log(log_list)

#LOG O/P
def update_log(names):
    log_output.config(state='normal')
    log_output.delete(1.0, tk.END)
    log_output.insert(tk.END, "\n".join(set(names)))
    log_output.config(state='disabled')

#SET'S THE TARGET VARIABLE
def set_target():
    global target_object
    target_object = entry.get().strip().lower()
    status_label.config(text=f" Target: {target_object}" if target_object else "Target cleared.")

def reset_target():
    global target_object
    target_object = None
    entry.delete(0, tk.END)
    status_label.config(text="Target reset.")

#THEAM INTERTER
def toggle_theme():
    global current_theme
    current_theme = "light" if current_theme == "dark" else "dark"
    apply_theme()

def apply_theme():
    dark = current_theme == "dark"
    root.config(bg="#1e1e2f" if dark else "#f0f0f0")
    side_panel.config(bg="#2d2d3d" if dark else "#e0e0e0")
    main_display.config(bg="#1e1e2f" if dark else "#ffffff")
    video_label.config(bg="#1e1e2f" if dark else "#ffffff")
    status_label.config(bg=main_display['bg'], fg="white" if dark else "black")
    log_output.config(bg="#2d2d3d" if dark else "#ffffff", fg="white" if dark else "black")

    for widget in side_panel.winfo_children():
        if isinstance(widget, tk.Button):
            widget.config(bg="#3c3f58" if dark else "#cccccc", fg="white" if dark else "black")
        elif isinstance(widget, tk.Label):
            widget.config(bg=side_panel['bg'], fg="white" if dark else "black")

#GUI
root = tk.Tk()
root.title("DetectorVision -- V HS2 Dashboard")
root.geometry("1500x1000")

# SLIDE PANEL
side_panel = tk.Frame(root, width=240)
side_panel.pack(side="left", fill="y")

main_display = tk.Frame(root)
main_display.pack(side="right", expand=True, fill="both")

def styled_button(text, command):
    btn = tk.Button(side_panel, text=text, command=command, font=("Arial", 11, "bold"), relief="flat", bd=0)
    btn.bind("<Enter>", lambda e: btn.config(bg="#404045"))
    btn.bind("<Leave>", lambda e: apply_theme())
    btn.pack(pady=5, padx=10, fill="x")
    return btn

#BUTTON AND LEBEL
tk.Label(side_panel, text="  DetectorVision", font=("Helvetica", 16, "bold")).pack(pady=(20, 10))
styled_button("  Start Live Feed", start_live_cam)
styled_button("  Stop Live Feed", stop_live_cam)
styled_button("  Detect From Video", detect_video)
styled_button("  Detect From Image", detect_image)

tk.Label(side_panel, text=" Targeted Object to Detect").pack(pady=(15, 5))
entry = tk.Entry(side_panel, font=("Arial", 12))
entry.pack(pady=5, padx=10, fill="x")
styled_button("Set Object", set_target)
styled_button("Reset Object", reset_target)
styled_button(" Change Theme", toggle_theme)
styled_button(" Close", root.destroy)

#O/P DISPLAY 
video_label = tk.Label(main_display)
video_label.pack(pady=10)

status_label = tk.Label(main_display, text="Ready.", font=("Arial", 12))
status_label.pack(pady=5)

log_output = tk.Text(main_display, height=6, width=50, font=("Consolas", 12))
log_output.pack(pady=10)
log_output.config(state='disabled')

apply_theme()
root.mainloop()


