"""
predict_apple.py
────────────────
Downloads the fruit-quality-r91st dataset from Roboflow, trains
YOLOv8n on it, then runs live detection from the GigE PoE camera.

Requirements:
    pip install roboflow ultralytics pyyaml

Usage:
    python3 predict_apple.py
"""

import cv2
import torch
import yaml
import os
from roboflow import Roboflow
from ultralytics import YOLO

# ── 1. Roboflow Dataset Config ─────────────────────────────────────────────────
RF_API_KEY   = "t0RbtP3ZQuikNZVPnZni"
RF_WORKSPACE = "defect-detection-k0lk0"
RF_PROJECT   = "fruit-quality-r91st"
RF_VERSION   = 1

# ── 2. Training Config ─────────────────────────────────────────────────────────
BASE_MODEL  = "yolov8n.pt"
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 8
PROJECT_DIR = "/home/orion/fruit_quality"
RUN_NAME    = "fruit_quality_v1"

# ── 3. Camera Config ───────────────────────────────────────────────────────────
CAMERA_NAME   = "03220069"
CAMERA_IP     = "192.168.4.100"
CAMERA_WIDTH  = 300
CAMERA_HEIGHT = 204              
DISPLAY_SIZE  = (640, 427)      # 3:2 aspect ratio of 3072x2048
CONF_THRESH   = 0.4

# ── 4. Keywords to determine good vs bad from class name ──────────────────────
GOOD_KEYWORDS = ["fresh", "good", "ripe", "healthy", "normal"]
BAD_KEYWORDS  = ["rotten", "bad", "spoil", "decay", "defect", "damaged", "unripe"]

def get_color(class_name: str):
    name = class_name.lower()
    if any(k in name for k in GOOD_KEYWORDS):
        return (0, 200, 0)      # green
    if any(k in name for k in BAD_KEYWORDS):
        return (0, 0, 255)      # red
    return (255, 165, 0)        # orange — unknown/other class

def is_good(class_name: str):
    return any(k in class_name.lower() for k in GOOD_KEYWORDS)

def is_bad(class_name: str):
    return any(k in class_name.lower() for k in BAD_KEYWORDS)

# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 1 — Download Dataset
# # ══════════════════════════════════════════════════════════════════════════════
# best_model_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")

# if os.path.exists(best_model_path):
#     print(f"Found existing trained model: {best_model_path}")
#     print("Skipping download and training — delete the weights folder to retrain.")
# else:
#     print("=" * 60)
#     print("Step 1: Downloading dataset from Roboflow...")
#     print("=" * 60)

#     rf = Roboflow(api_key=RF_API_KEY)
#     project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
#     dataset = project.version(RF_VERSION).download("yolov8")

#     DATA_YAML = os.path.join(dataset.location, "data.yaml")
#     print(f"\nDataset downloaded to: {dataset.location}")

#     # Print classes so you know what the model will detect
#     with open(DATA_YAML, 'r') as f:
#         data_info = yaml.safe_load(f)
#     print(f"\nDataset classes ({data_info.get('nc')} total):")
#     for i, name in enumerate(data_info.get('names', [])):
#         tag = "GOOD" if is_good(name) else ("BAD" if is_bad(name) else "OTHER")
#         print(f"  [{i}] {name:30s} → {tag}")

#     # ── STEP 2 — Train ─────────────────────────────────────────────────────────
#     print("\n" + "=" * 60)
#     print("Step 2: Training YOLOv8n...")
#     print("=" * 60)

#     model = YOLO(BASE_MODEL)
#     model.train(
#         data=DATA_YAML,
#         epochs=EPOCHS,
#         imgsz=IMG_SIZE,
#         batch=BATCH_SIZE,
#         device=0 if torch.cuda.is_available() else 'cpu',
#         project=PROJECT_DIR,
#         name=RUN_NAME,
#         patience=15,
#         save=True,
#         plots=True,
#         val=True,
#         augment=True,
#         cache=True,
#         workers=2,
#         cos_lr=True,
#         verbose=True,
#     )
#     print(f"\nTraining complete. Best model: {best_model_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Load Trained Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: Loading trained model for live inference...")
print("=" * 60)

model = YOLO('/home/orion/robo/fv/Fruit-Quality-1/best.pt')
class_names = model.names       # {0: 'class_a', 1: 'class_b', ...}

print(f"Model classes:")
for idx, name in class_names.items():
    tag = "GOOD" if is_good(name) else ("BAD" if is_bad(name) else "OTHER")
    print(f"  [{idx}] {name:30s} → {tag}")

device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Open Camera
# ══════════════════════════════════════════════════════════════════════════════
pipeline = (
    f"aravissrc camera-name=\"{CAMERA_NAME}\" ! "
    f"video/x-raw,format=GRAY8,width={CAMERA_WIDTH},height={CAMERA_HEIGHT} ! "
    "queue ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! " 
    "appsink drop=true max-buffers=1 sync=false"
)
 
print(f"\nConnecting to {CAMERA_IP} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
 
if not cap.isOpened():
    print("CRITICAL: Could not open camera pipeline.")
    print(f"  Check: ping {CAMERA_IP}")
    print(f"  Check: arv-tool-0.8 detect")
    exit(1)
 
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Capture resolution confirmed: {actual_w}x{actual_h}")
 
cv2.namedWindow('Fruit Quality Inspector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fruit Quality Inspector', DISPLAY_SIZE[0], DISPLAY_SIZE[1])
 
print("\nStarting live detection. Press 'q' to quit.")
 
# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Live Inference Loop
# ══════════════════════════════════════════════════════════════════════════════
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
 
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            device=device,
            verbose=False
        )
 
        display = cv2.resize(frame, DISPLAY_SIZE)
        sx = DISPLAY_SIZE[0] / CAMERA_WIDTH
        sy = DISPLAY_SIZE[1] / CAMERA_HEIGHT
 
        good_count = bad_count = other_count = 0
 
        for box in results[0].boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = class_names[cls_id]
            color      = get_color(class_name)
 
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * sx); y1 = int(y1 * sy)
            x2 = int(x2 * sx); y2 = int(y2 * sy)
 
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 
            if is_good(class_name):
                good_count += 1
            elif is_bad(class_name):
                bad_count += 1
            else:
                other_count += 1
 
        # Summary bar
        cv2.rectangle(display, (0, 0), (DISPLAY_SIZE[0], 30), (0, 0, 0), -1)
        summary = (f"Good: {good_count}   "
                   f"Bad: {bad_count}   "
                   f"Total: {good_count + bad_count + other_count}")
        cv2.putText(display, summary, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
 
        cv2.imshow('Fruit Quality Inspector', display)
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
cap.release()
cv2.destroyAllWindows()
