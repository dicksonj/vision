import cv2
import torch
import os
from ultralytics import YOLO

# ── 1. Model Config ────────────────────────────────────────────────────────────
MODEL_PATH = '/home/orion/robo/fv/Fruit-Quality-1/best.pt'

# ── 2. Camera Config ───────────────────────────────────────────────────────────
CAMERA_IP     = "192.168.4.100"
CAMERA_WIDTH  = 3072
CAMERA_HEIGHT = 2048
DISPLAY_SIZE  = (640, 427)      # 3:2 aspect ratio of 3072x2048
CONF_THRESH   = 0.4
IMG_SIZE      = 640

# ── 3. Keywords to determine good vs bad from class name ──────────────────────
GOOD_KEYWORDS = ["fresh", "good", "ripe", "healthy", "normal"]
BAD_KEYWORDS  = ["rotten", "bad", "spoil", "decay", "defect", "damaged", "unripe"]

def get_color(class_name: str):
    name = class_name.lower()
    if any(k in name for k in GOOD_KEYWORDS):
        return (0, 200, 0)      # green
    if any(k in name for k in BAD_KEYWORDS):
        return (0, 0, 255)      # red
    return (255, 165, 0)        # orange — unknown class

def is_good(class_name: str):
    return any(k in class_name.lower() for k in GOOD_KEYWORDS)

def is_bad(class_name: str):
    return any(k in class_name.lower() for k in BAD_KEYWORDS)

# ── 4. Load Model ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading model...")
print("=" * 60)

model       = YOLO(MODEL_PATH)
class_names = model.names
device      = 0 if torch.cuda.is_available() else 'cpu'

print(f"Model classes:")
for idx, name in class_names.items():
    tag = "GOOD" if is_good(name) else ("BAD" if is_bad(name) else "OTHER")
    print(f"  [{idx}] {name:30s} → {tag}")
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ── 5. GStreamer Pipeline ──────────────────────────────────────────────────────
# Use IP directly — avoids quoted camera name issues with OpenCV GStreamer parser
pipeline = (
    f"aravissrc camera-name={CAMERA_IP} ! "
    f"video/x-raw,format=GRAY8,width={CAMERA_WIDTH},height={CAMERA_HEIGHT} ! "
    "videoconvert ! "
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

# ── 6. Live Inference Loop ─────────────────────────────────────────────────────
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Camera outputs GRAY8 — convert to BGR so YOLO gets a 3-channel input
        # Without this, frame.shape = (2048, 3072) and YOLO will error
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            device=device,
            verbose=False
        )

        # Scale display frame down from 3072x2048
        display = cv2.resize(frame, DISPLAY_SIZE)
        sx = DISPLAY_SIZE[0] / CAMERA_WIDTH
        sy = DISPLAY_SIZE[1] / CAMERA_HEIGHT

        good_count = bad_count = other_count = 0

        for box in results[0].boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = class_names[cls_id]
            color      = get_color(class_name)

            # Scale bounding box coords to display size
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

        # Summary bar at top
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