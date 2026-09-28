"""
predict_apple.py
────────────────
Live apple quality detection using a YOLOv8 model trained on
fresh/rotten apple data, streaming from the GigE PoE camera.

Requirements:
    pip install ultralytics

Usage:
    1. Run train_apple.py first to get best.pt
    2. Update MODEL_PATH below with the path printed by train_apple.py
    3. Run:  python3 predict_apple.py
"""

import cv2
import torch
from ultralytics import YOLO

# ── 1. Constants ───────────────────────────────────────────────────────────────
# Update this to the best.pt path printed at the end of train_apple.py
MODEL_PATH    = "/home/orion/apple_quality/apple_fresh_rotten_v1/weights/best.pt"

CAMERA_NAME   = "The Imaging Source Europe GmbH-DMK 33GX178-03220069"
CAMERA_IP     = "192.168.4.100"
CAMERA_WIDTH  = 3072
CAMERA_HEIGHT = 2048
DISPLAY_SIZE  = (640, 427)      # 3:2 aspect ratio scaled for display
INFER_SIZE    = 640             # must match IMG_SIZE used during training
CONF_THRESH   = 0.4             # raise if too many false positives

# ── 2. Class Colors ────────────────────────────────────────────────────────────
# These map class names (from your Roboflow dataset) to display colors
# Adjust keys to match whatever names your dataset used
CLASS_COLORS = {
    "fresh":         (0,   200, 0),     # green
    "rotten":        (0,   0,   255),   # red
    "fresh_apple":   (0,   200, 0),
    "rotten_apple":  (0,   0,   255),
    "good":          (0,   200, 0),
    "bad":           (0,   0,   255),
}
DEFAULT_GOOD_COLOR = (0,   200, 0)
DEFAULT_BAD_COLOR  = (0,   0,   255)

def get_color(class_name: str) -> tuple:
    """Return color based on whether class name implies fresh or rotten."""
    name = class_name.lower()
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    # Fallback: infer from keywords
    if any(w in name for w in ["fresh", "good", "ripe", "healthy"]):
        return DEFAULT_GOOD_COLOR
    if any(w in name for w in ["rotten", "bad", "spoil", "decay", "defect"]):
        return DEFAULT_BAD_COLOR
    return (255, 165, 0)    # orange for unknown classes

# ── 3. Load Model ──────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

class_names = model.names   # dict of {index: class_name} from training
print(f"Model classes: {class_names}")

device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ── 4. GStreamer Pipeline ──────────────────────────────────────────────────────
pipeline = (
    f"aravissrc camera-name=\"{CAMERA_NAME}\" ! "
    f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT} ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

print(f"Connecting to {CAMERA_IP} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("CRITICAL: Could not open camera pipeline.")
    print(f"  Check: ping {CAMERA_IP}")
    print(f"  Check: arv-tool-0.8 detect")
    exit(1)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Capture resolution confirmed: {actual_w}x{actual_h}")

# ── 5. Window Setup ────────────────────────────────────────────────────────────
cv2.namedWindow('Apple Quality Inspector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Apple Quality Inspector', DISPLAY_SIZE[0], DISPLAY_SIZE[1])

print("Starting detection. Press 'q' to quit.")

# ── 6. Inference Loop ──────────────────────────────────────────────────────────
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run inference
        results = model.predict(
            source=frame,
            imgsz=INFER_SIZE,
            conf=CONF_THRESH,
            device=device,
            verbose=False
        )

        # ── Draw custom annotations ────────────────────────────────────────────
        display = cv2.resize(frame, DISPLAY_SIZE)

        # Scale factor from capture res to display res
        sx = DISPLAY_SIZE[0] / CAMERA_WIDTH
        sy = DISPLAY_SIZE[1] / CAMERA_HEIGHT

        fresh_count  = 0
        rotten_count = 0

        for box in results[0].boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = class_names[cls_id]
            color      = get_color(class_name)

            # Scale bounding box to display size
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(x1*sx), int(y1*sy),
                               int(x2*sx), int(y2*sy))

            # Draw box and label
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 6),
                          (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Count fresh vs rotten
            name_lower = class_name.lower()
            if any(w in name_lower for w in ["fresh", "good", "ripe", "healthy"]):
                fresh_count += 1
            elif any(w in name_lower for w in ["rotten", "bad", "spoil", "decay", "defect"]):
                rotten_count += 1

        # ── Summary bar at top ─────────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (DISPLAY_SIZE[0], 30), (0, 0, 0), -1)
        summary = f"Fresh: {fresh_count}   Rotten: {rotten_count}   Total: {fresh_count + rotten_count}"
        cv2.putText(display, summary, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Apple Quality Inspector', display)

        # Clear GPU cache to prevent memory leakage on Jetson
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
