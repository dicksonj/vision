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
CAMERA_NAME   = "The Imaging Source Europe GmbH-DMK 33GX178-03220069"
CAMERA_IP     = "192.168.4.100"
CAMERA_WIDTH  = 3072
CAMERA_HEIGHT = 2048
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
 
# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download Dataset
# ══════════════════════════════════════════════════════════════════════════════
best_model_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
 
if os.path.exists(best_model_path):
    print(f"Found existing trained model: {best_model_path}")
    print("Skipping download and training — delete the weights folder to retrain.")
else:
    print("=" * 60)
    print("Step 1: Downloading dataset from Roboflow...")
    print("=" * 60)
 
    rf = Roboflow(api_key=RF_API_KEY)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    dataset = project.version(RF_VERSION).download("yolov8")
 
    DATA_YAML = os.path.join(dataset.location, "data.yaml")
    print(f"\nDataset downloaded to: {dataset.location}")
 
    # Print classes so you know what the model will detect
    with open(DATA_YAML, 'r') as f:
        data_info = yaml.safe_load(f)
    print(f"\nDataset classes ({data_info.get('nc')} total):")
    for i, name in enumerate(data_info.get('names', [])):
        tag = "GOOD" if is_good(name) else ("BAD" if is_bad(name) else "OTHER")
        print(f"  [{i}] {name:30s} → {tag}")
 
    # ── STEP 2 — Train ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Training YOLOv8n...")
    print("=" * 60)
 
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=15,
        save=True,
        plots=True,
        val=True,
        augment=True,
        cache=True,
        workers=2,
        cos_lr=True,
        verbose=True,
    )
    print(f"\nTraining complete. Best model: {best_model_path}")
