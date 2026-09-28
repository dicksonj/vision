import cv2
import time
import threading
import numpy as np
import streamlit as st
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
#  1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDS — Pedestrian Surveillance (No PyTorch)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  2. CUSTOM CSS (Dark Industrial Aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    background-color: #0a0c10;
    color: #c8d0dc;
    font-family: 'Barlow Condensed', sans-serif;
}
.stApp { background-color: #0a0c10; }

[data-testid="stSidebar"] {
    background: #0d1017;
    border-right: 1px solid #1e2530;
}
[data-testid="stSidebar"] * { color: #c8d0dc !important; }

h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.08em; }

.metric-card {
    background: #111520;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 18px 20px 14px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00e5ff, #0050ff);
}
.metric-card.danger::before { background: linear-gradient(90deg, #ff2244, #ff6600); }
.metric-card.warn::before   { background: linear-gradient(90deg, #ffcc00, #ff8800); }
.metric-card.ok::before     { background: linear-gradient(90deg, #00ff99, #00ccaa); }

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #556070;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 30px;
    font-weight: bold;
    color: #e8f0ff;
    line-height: 1;
}

.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.1em;
    font-weight: bold;
    text-transform: uppercase;
}
.status-online  { background: #002a1a; color: #00ff99; border: 1px solid #00ff9933; }
.status-alert   { background: #2a0008; color: #ff2244; border: 1px solid #ff224433; animation: pulse 0.8s infinite alternate; }
.status-offline { background: #1a1a1a; color: #556070; border: 1px solid #33333355; }

@keyframes pulse { from { opacity: 1; } to { opacity: 0.5; } }

.panel-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: #445060;
    text-transform: uppercase;
    padding: 6px 0 8px;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 10px;
}
.panel-header span { color: #00e5ff; margin-right: 8px; }

hr { border-color: #1e2530 !important; }

.log-entry {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    padding: 4px 0;
    border-bottom: 1px solid #111820;
    color: #445060;
}
.log-entry .ts  { color: #223040; margin-right: 8px; }
.log-entry.warn { color: #ffcc00; }
.log-entry.crit { color: #ff2244; }
.log-entry.ok   { color: #00ff99; }

.stButton > button {
    background: #111520 !important;
    color: #00e5ff !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.1em !important;
    width: 100%;
}
.stButton > button:hover { border-color: #00e5ff !important; }

[data-testid="stImage"] img {
    border: 1px solid #1e2530;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  3. CONFIGURATION & FISHEYE PARAMS
# ─────────────────────────────────────────────
ONNX_MODEL_PATH    = "yolov8n.onnx"
USB_CAMERA_INDEX   = 0           # Standard USB webcam index
CONF_THRESH        = 0.40        # Confidence threshold
NMS_THRESH         = 0.45        # Non-maximum suppression threshold
PERSON_CLASS_ID    = 0           # COCO Class index for 'person'
DISPLAY_SIZE       = (640, 480)
INPUT_WIDTH        = 640
INPUT_HEIGHT       = 640

# Fisheye Camera Matrix & Distortion Coefficients
K = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
D = np.array([[-0.28], [0.08], [-0.01], [0.00]])

# ─────────────────────────────────────────────
#  4. SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────
def _init_state():
    defaults = dict(
        running=False,
        grabber=None,
        pedestrian_count=0,
        primary_direction="STATIONARY",
        total_detections_session=0,
        raw_frame=None,
        pred_frame=None,
        centroids=deque(maxlen=20),
        log=deque(maxlen=50),
        status="OFFLINE"
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────
#  5. THREADED USB FRAME GRABBER
# ─────────────────────────────────────────────
class USBFrameGrabber(threading.Thread):
    def __init__(self, camera_index=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_latest(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

# ─────────────────────────────────────────────
#  6. LOAD OPENCV DNN MODEL (NO PYTORCH)
# ─────────────────────────────────────────────
@st.cache_resource
def load_onnx_model():
    net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
    # Enable CUDA backend if available in OpenCV build
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

try:
    net = load_onnx_model()
except Exception as e:
    st.error(f"Failed to load ONNX model '{ONNX_MODEL_PATH}'. Ensure file exists. Error: {e}")
    st.stop()

# ─────────────────────────────────────────────
#  7. HELPER FUNCTIONS
# ─────────────────────────────────────────────
def _log(msg: str, kind: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.appendleft({"ts": ts, "msg": msg, "kind": kind})

def start_camera():
    grabber = USBFrameGrabber(USB_CAMERA_INDEX)
    grabber.start()

    st.session_state.grabber = grabber
    st.session_state.running = True
    st.session_state.status = "ONLINE"
    _log("Fisheye USB stream operational", kind="ok")

def stop_camera():
    if st.session_state.grabber:
        st.session_state.grabber.stop()
    st.session_state.running = False
    st.session_state.status = "OFFLINE"
    st.session_state.grabber = None
    _log("Surveillance system offline", kind="warn")

def reset_counters():
    st.session_state.pedestrian_count = 0
    st.session_state.primary_direction = "STATIONARY"
    st.session_state.total_detections_session = 0
    st.session_state.centroids.clear()
    st.session_state.log.clear()
    _log("Telemetry reset", kind="ok")

def undistort_fisheye(frame):
    """Corrects distortion from wide-angle/fisheye USB lenses."""
    h, w = frame.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def calculate_direction(points):
    """Calculates directional displacement vector across historical centroids."""
    if len(points) < 5:
        return "STATIONARY"
    
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    dx = end_x - start_x
    dy = end_y - start_y

    if abs(dx) < 15 and abs(dy) < 15:
        return "STATIONARY"
    
    if abs(dx) > abs(dy):
        return "EAST 👉" if dx > 0 else "WEST 👈"
    else:
        return "SOUTH ⬇️" if dy > 0 else "NORTH ⬆️"

def post_process_yolo_onnx(outputs, orig_shape):
    """Parses raw ONNX outputs using NumPy (No PyTorch)."""
    h_orig, w_orig = orig_shape[:2]
    
    # ONNX shape typically: [1, 84, 8400] -> transpose to [8400, 84]
    predictions = np.squeeze(outputs[0]).T
    
    boxes, confidences, class_ids = [], [], []
    x_factor = w_orig / INPUT_WIDTH
    y_factor = h_orig / INPUT_HEIGHT

    for row in predictions:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        confidence = classes_scores[class_id]

        # Only extract 'person' class predictions above threshold
        if class_id == PERSON_CLASS_ID and confidence >= CONF_THRESH:
            x, y, w, h = row[0], row[1], row[2], row[3]
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # OpenCV Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append((boxes[i], confidences[i]))
            
    return final_boxes

# ─────────────────────────────────────────────
#  8. PIPELINE INFERENCE STEP (OPENCV DNN)
# ─────────────────────────────────────────────
def process_frame():
    s = st.session_state
    if not s.running or s.grabber is None:
        return

    frame = s.grabber.get_latest()
    if frame is None:
        return

    # Undistort wide-angle lens
    corrected_frame = undistort_fisheye(frame)

    # Pre-process frame for ONNX network
    blob = cv2.dnn.blobFromImage(
        corrected_frame, 
        1/255.0, 
        (INPUT_WIDTH, INPUT_HEIGHT), 
        swapRB=True, 
        crop=False
    )
    net.setInput(blob)
    outputs = net.forward()

    # Parse predictions
    detections = post_process_yolo_onnx(outputs, corrected_frame.shape)

    s.raw_frame = cv2.resize(corrected_frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
    pred = s.raw_frame.copy()

    h_orig, w_orig = corrected_frame.shape[:2]
    sx = DISPLAY_SIZE[0] / w_orig
    sy = DISPLAY_SIZE[1] / h_orig

    active_pedestrians = len(detections)

    if active_pedestrians > 0:
        avg_x, avg_y = 0, 0
        for (box, conf) in detections:
            x, y, w, h = box
            
            # Map coordinates to display scaling
            rx1, ry1 = int(x * sx), int(y * sy)
            rx2, ry2 = int((x + w) * sx), int((y + h) * sy)

            cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            avg_x += cx
            avg_y += cy

            # Render Bounding Box
            cv2.rectangle(pred, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            label = f"PEDESTRIAN {conf:.0%}"
            cv2.putText(pred, label, (rx1, ry1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Log average centroid movement to track vector direction
        avg_x //= active_pedestrians
        avg_y //= active_pedestrians
        s.centroids.append((avg_x, avg_y))
        
        direction = calculate_direction(s.centroids)
        s.primary_direction = direction

        # Draw trajectory vector line
        if len(s.centroids) > 1:
            pts = np.array(s.centroids, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(pred, [pts], isClosed=False, color=(0, 229, 255), thickness=2)

    s.pedestrian_count = active_pedestrians
    s.total_detections_session += active_pedestrians

    if active_pedestrians > 0:
        s.status = "ALERT — HUMAN DETECTED"
        _log(f"ALERT: {active_pedestrians} pedestrian(s) detected [{s.primary_direction}]", kind="crit")
    else:
        s.status = "ONLINE"
        s.primary_direction = "NONE"

    s.pred_frame = pred

# ─────────────────────────────────────────────
#  9. SIDEBAR LAYOUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding:10px 0 20px'>
            <div style='font-family:Share Tech Mono,monospace;font-size:10px;letter-spacing:0.2em;color:#445060;'>OPTICAL SURVEILLANCE</div>
            <div style='font-family:Barlow Condensed,sans-serif;font-size:26px;font-weight:800;letter-spacing:0.05em;color:#e8f0ff;line-height:1.1;'>
                PEDESTRIAN<br>DETECTOR
            </div>
            <div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#334055;letter-spacing:0.15em;margin-top:4px;'>
                OPENCV DNN · ONNX ENGINE
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    s = st.session_state
    badge_cls = "status-alert" if "ALERT" in s.status else ("status-online" if s.status == "ONLINE" else "status-offline")
    st.markdown(f'<div class="status-badge {badge_cls}">{s.status}</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Telemetry Cards
    st.markdown(f"""
    <div class="metric-card {"danger" if s.pedestrian_count > 0 else "ok"}">
        <div class="metric-label">ACTIVE PEDESTRIANS</div>
        <div class="metric-value">{s.pedestrian_count}</div>
    </div>
    <div class="metric-card warn">
        <div class="metric-label">PRIMARY TRAJECTORY</div>
        <div class="metric-value" style="font-size:22px;">{s.primary_direction}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">TOTAL ACCUMULATED</div>
        <div class="metric-value">{s.total_detections_session:,}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ START" if not s.running else "⏹ STOP"):
            if not s.running:
                start_camera()
            else:
                stop_camera()
            st.rerun()
    with col2:
        if st.button("↺ RESET"):
            reset_counters()
            st.rerun()

    st.markdown("---")

    # Realtime Event Log
    st.markdown('<div class="metric-label" style="margin-bottom:8px">SURVEILLANCE EVENT LOG</div>', unsafe_allow_html=True)
    log_html = ""
    for entry in list(s.log)[:20]:
        log_html += f'<div class="log-entry {entry["kind"]}"><span class="ts">{entry["ts"]}</span>{entry["msg"]}</div>'
    if not log_html:
        log_html = '<div class="log-entry"><span class="ts">--:--:--</span>System idle. No detections.</div>'
    st.markdown(log_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  10. MAIN DASHBOARD VIEW
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:4px;padding-bottom:10px;border-bottom:1px solid #1e2530;'>
    <span style='font-family:Barlow Condensed,sans-serif;font-size:28px;font-weight:800;letter-spacing:0.08em;color:#e8f0ff;'>PEDESTRIAN MONITORING CONSOLE</span>
    <span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#334055;letter-spacing:0.15em;'>NO-PYTORCH OPENCV ONNX PIPELINE</span>
</div>
""", unsafe_allow_html=True)

# Process active frame
process_frame()

col_prev, col_pred = st.columns(2, gap="medium")

# Raw Stream
with col_prev:
    st.markdown('<div class="panel-header"><span>◈</span>UNDISTORTED RAW FISHEYE FEED</div>', unsafe_allow_html=True)
    prev_placeholder = st.empty()
    if s.raw_frame is not None:
        prev_placeholder.image(cv2.cvtColor(s.raw_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        prev_placeholder.markdown("""
        <div style='background:#0d1017;border:1px solid #1e2530;border-radius:4px;height:350px;display:flex;align-items:center;justify-content:center;font-family:Share Tech Mono,monospace;font-size:12px;color:#334055;'>
            USB CAMERA DISCONNECTED
        </div>
        """, unsafe_allow_html=True)

# Prediction Overlay Stream
with col_pred:
    st.markdown('<div class="panel-header"><span>◈</span>OPENCV DNN INFERENCE & VECTOR TRACKING</div>', unsafe_allow_html=True)
    pred_placeholder = st.empty()
    if s.pred_frame is not None:
        pred_placeholder.image(cv2.cvtColor(s.pred_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        pred_placeholder.markdown("""
        <div style='background:#0d1017;border:1px solid #1e2530;border-radius:4px;height:350px;display:flex;align-items:center;justify-content:center;font-family:Share Tech Mono,monospace;font-size:12px;color:#334055;'>
            INFERENCE ENGINE IDLE
        </div>
        """, unsafe_allow_html=True)

# Legend
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
leg1, leg2, leg3 = st.columns(3)
legends = [
    ("#ff2244", "HUMAN DETECTED", "Person detected via ONNX model predictions."),
    ("#00e5ff", "VECTOR TRAJECTORY", "Centroid directional path over last 20 frames."),
    ("#00ff99", "MONITORED ZONE", "Undistorted fisheye region monitored by OpenCV DNN."),
]
for col, (clr, title, desc) in zip([leg1, leg2, leg3], legends):
    with col:
        st.markdown(f"""
        <div style='border-left:3px solid {clr};padding:6px 10px;background:#0d1017;border-radius:0 4px 4px 0;'>
            <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:{clr};letter-spacing:0.1em;'>{title}</div>
            <div style='font-size:11px;color:#445060;margin-top:2px;font-family:Barlow Condensed,sans-serif;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Loop trigger
if s.running:
    time.sleep(0.01)
    st.rerun()