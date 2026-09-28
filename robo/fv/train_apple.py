"""
train_apple.py
──────────────
Downloads a fresh/rotten apple dataset from Roboflow and fine-tunes
YOLOv8n on it for good (fresh) vs bad (rotten) apple detection.

Requirements:
    pip install roboflow ultralytics

Usage:
    1. Create a free account at https://roboflow.com
    2. Go to https://universe.roboflow.com/search?q=apple+fresh+rotten
    3. Pick a dataset, click "Download" → choose YOLOv8 format
    4. Copy your API key from https://app.roboflow.com/settings/api
    5. Fill in RF_API_KEY, RF_WORKSPACE, RF_PROJECT, RF_VERSION below
    6. Run:  python3 train_apple.py
"""

from roboflow import Roboflow
from ultralytics import YOLO
import os

# ── 1. Roboflow Config ─────────────────────────────────────────────────────────
# Fill these in from your Roboflow project page
RF_API_KEY  = "t0RbtP3ZQuikNZVPnZni"        # https://app.roboflow.com/settings/api
RF_WORKSPACE = "jd_dev"     # e.g. "john-smith-abc12"
RF_PROJECT   = "fresh_fruits"       # e.g. "apple-freshness"
RF_VERSION   = 1                         # dataset version number

# ── 2. Training Config ─────────────────────────────────────────────────────────
BASE_MODEL   = "yolov8n.pt"             # nano — best for Jetson Orin real-time
EPOCHS       = 50                        # increase to 100 for better accuracy
IMG_SIZE     = 640                       # match your inference script
BATCH_SIZE   = 8                         # reduce to 4 if Jetson runs out of memory
DEVICE       = 0                         # 0 = first GPU (Jetson Orin GPU)
PROJECT_DIR  = "/home/orion/robo/fv"
RUN_NAME     = "fresh_fruits_v1"

# ── 3. Download Dataset from Roboflow ─────────────────────────────────────────
print("=" * 60)
print("Step 1: Downloading dataset from Roboflow...")
print("=" * 60)

rf = Roboflow(api_key=RF_API_KEY)
project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
dataset = project.version(RF_VERSION).download("yolov8")

# dataset.location contains the path to downloaded data.yaml
DATA_YAML = os.path.join(dataset.location, "data.yaml")
print(f"\nDataset downloaded to: {dataset.location}")
print(f"data.yaml path: {DATA_YAML}")

# ── 4. Print Class Info ────────────────────────────────────────────────────────
# So you know which class index = fresh vs rotten
import yaml
with open(DATA_YAML, 'r') as f:
    data_info = yaml.safe_load(f)
print(f"\nDataset classes: {data_info.get('names', 'unknown')}")
print(f"Number of classes: {data_info.get('nc', 'unknown')}")

# ── 5. Train ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Starting YOLOv8 training...")
print("=" * 60)

model = YOLO(BASE_MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    patience=15,            # stop early if no improvement for 15 epochs
    save=True,              # save best.pt and last.pt
    plots=True,             # save training plots (confusion matrix, PR curve)
    val=True,               # run validation after each epoch
    augment=True,           # use data augmentation (helps with small datasets)
    cache=True,             # cache images in RAM for faster training on Jetson
    workers=2,              # data loader workers (keep low on Jetson)
    cos_lr=True,            # cosine learning rate scheduler
    verbose=True,
)

# ── 6. Validate and Report ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Running final validation...")
print("=" * 60)

metrics = model.val()
print(f"\nmAP50:     {metrics.box.map50:.3f}")
print(f"mAP50-95:  {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall:    {metrics.box.mr:.3f}")

# ── 7. Export Best Model Path ──────────────────────────────────────────────────
best_model = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
print(f"\n{'=' * 60}")
print(f"Training complete!")
print(f"Best model saved at:\n  {best_model}")
print(f"\nUse this path in predict_apple.py:")
print(f'  MODEL_PATH = "{best_model}"')
print("=" * 60)
