import cv2
import torch
import time
import threading
import streamlit as st
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FQA — Fruit Quality Automation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark industrial aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap');

/* ── global ── */
html, body, [class*="css"] {
    background-color: #0a0c10;
    color: #c8d0dc;
    font-family: 'Barlow Condensed', sans-serif;
}
.stApp { background-color: #0a0c10; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1017;
    border-right: 1px solid #1e2530;
}
[data-testid="stSidebar"] * { color: #c8d0dc !important; }

/* ── headings ── */
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.08em; }

/* ── metric cards ── */
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
    font-size: 32px;
    font-weight: bold;
    color: #e8f0ff;
    line-height: 1;
}

/* ── status badge ── */
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
.status-focused    { background: #002a1a; color: #00ff99; border: 1px solid #00ff9933; }
.status-distracted { background: #2a0008; color: #ff2244; border: 1px solid #ff224433;
                     animation: pulse 0.8s infinite alternate; }
.status-offline    { background: #1a1a1a; color: #556070; border: 1px solid #33333355; }

@keyframes pulse { from { opacity: 1; } to { opacity: 0.5; } }

/* ── video panel ── */
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

/* ── divider ── */
hr { border-color: #1e2530 !important; }

/* ── log ── */
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

/* ── buttons ── */
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

/* ── image frame ── */
[data-testid="stImage"] img {
    border: 1px solid #1e2530;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH    = '/home/orion/robo/fv/Fruit-Quality-1/best.pt'
CAMERA_IP     = "192.168.4.100"
CAMERA_WIDTH  = 3072
CAMERA_HEIGHT = 2048
DISPLAY_SIZE  = (640, 427)
CONF_THRESH   = 0.4
IMG_SIZE      = 640

GOOD_KEYWORDS = ["fresh", "good", "ripe", "healthy", "normal"]
BAD_KEYWORDS  = ["rotten", "bad", "spoil", "decay", "defect", "damaged", "unripe"]

def get_color(class_name: str):
    name = class_name.lower()
    if any(k in name for k in GOOD_KEYWORDS):
        return (0, 200, 0)      # Green (BGR)
    if any(k in name for k in BAD_KEYWORDS):
        return (0, 0, 255)      # Red (BGR)
    return (255, 165, 0)        # Orange

def is_good(class_name: str):
    return any(k in class_name.lower() for k in GOOD_KEYWORDS)

def is_not_rotten(class_name: str):
    """Any fruit that does NOT match a bad/rotten keyword."""
    return not any(k in class_name.lower() for k in BAD_KEYWORDS)

def is_bad(class_name: str):
    return any(k in class_name.lower() for k in BAD_KEYWORDS)

# ─────────────────────────────────────────────
#  SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────
def _init_state():
    defaults = dict(
        running=False,
        grabber=None,
        cap=None,
        good_count=0,
        bad_count=0,
        total_count=0,
        raw_frame=None,
        pred_frame=None,
        log=deque(maxlen=50),
        status="OFFLINE"
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────
#  FRAME GRABBER THREAD
# ─────────────────────────────────────────────
class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap     = cap
        self.frame   = None
        self.lock    = threading.Lock()
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

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    model = YOLO(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else 'cpu'
    return model, device

try:
    model, device = load_yolo_model()
    class_names = model.names
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}. Error: {e}")
    st.stop()

# ─────────────────────────────────────────────
#  START / STOP CONTROLS
# ─────────────────────────────────────────────
def _log(msg: str, kind: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.appendleft({"ts": ts, "msg": msg, "kind": kind})

def start_camera():
    pipeline = (
        f"aravissrc camera-name={CAMERA_IP} ! "
        f"video/x-raw,format=GRAY8,width={CAMERA_WIDTH},height={CAMERA_HEIGHT} ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        st.error("❌ Cannot open camera pipeline — check connections.")
        _log("Connection failed", kind="crit")
        return

    grabber = FrameGrabber(cap)
    grabber.start()

    st.session_state.cap = cap
    st.session_state.grabber = grabber
    st.session_state.running = True
    st.session_state.status = "ONLINE"
    _log("Inspection line operational", kind="ok")

def stop_camera():
    if st.session_state.grabber:
        st.session_state.grabber.stop()
    if st.session_state.cap:
        st.session_state.cap.release()
    st.session_state.running = False
    st.session_state.status = "OFFLINE"
    st.session_state.grabber = None
    st.session_state.cap = None
    _log("Inspection line stopped", kind="warn")

def reset_counters():
    st.session_state.good_count = 0
    st.session_state.bad_count = 0
    st.session_state.total_count = 0
    st.session_state.log.clear()
    _log("Production telemetry reset", kind="ok")

# ─────────────────────────────────────────────
#  PIPELINE PROCESSING STEP
# ─────────────────────────────────────────────
def process_frame():
    s = st.session_state
    if not s.running or s.grabber is None:
        return

    frame = s.grabber.get_latest()
    if frame is None:
        return

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Perform YOLO Inference
    with torch.no_grad():
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            device=device,
            verbose=False
        )

    # Resize background frames for performance UI scaling
    s.raw_frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
    pred = s.raw_frame.copy()
    
    sx = DISPLAY_SIZE[0] / CAMERA_WIDTH
    sy = DISPLAY_SIZE[1] / CAMERA_HEIGHT

    local_good = local_bad = local_other = 0

    for box in results[0].boxes:
        cls_id     = int(box.cls[0])
        conf       = float(box.conf[0])
        class_name = class_names[cls_id]
        color      = get_color(class_name)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(x1 * sx); y1 = int(y1 * sy)
        x2 = int(x2 * sx); y2 = int(y2 * sy)

        # Draw box and label overlays
        cv2.rectangle(pred, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(pred, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(pred, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if is_bad(class_name):
            local_bad += 1
        elif is_not_rotten(class_name):
            local_good += 1
        else:
            local_other += 1

    # Update persistent telemetry counters if new classifications hit
    s.good_count += local_good
    s.bad_count += local_bad
    s.total_count += (local_good + local_bad + local_other)

    if local_bad > 0:
        _log(f"Defect spotted: {local_bad} anomalies flagged", kind="crit")
    elif local_good > 0:
        _log(f"Non-rotten fruit: +{local_good} accepted", kind="ok")

    s.pred_frame = pred
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────
#  SIDEBAR LAYOUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding:10px 0 20px'>
            <div style='font-family:Share Tech Mono,monospace;font-size:10px;
                        letter-spacing:0.2em;color:#445060;'>AUTOMATION</div>
            <div style='font-family:Barlow Condensed,sans-serif;font-size:26px;
                        font-weight:800;letter-spacing:0.05em;color:#e8f0ff;line-height:1.1;'>
                FRUIT QUALITY<br>INSPECTOR
            </div>
            <div style='font-family:Share Tech Mono,monospace;font-size:9px;
                        color:#334055;letter-spacing:0.15em;margin-top:4px;'>
                ROBO EDGE · YOLO CORE
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    s = st.session_state
    badge_cls = "status-focused" if s.status == "ONLINE" else "status-offline"
    st.markdown(f'<div class="status-badge {badge_cls}">{s.status}</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Telemetry Widget Cards
    for label, val, cls in [
        ("NON-ROTTEN FRUIT COUNT", s.good_count, "ok"),
        ("BAD OBJECTS COUNT",  s.bad_count,  "danger"),
        ("TOTAL PROCESSING",   s.total_count, ""),
    ]:
        st.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # App Toggling Triggers
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

    # Realtime Line Logging
    st.markdown('<div class="metric-label" style="margin-bottom:8px">SYSTEM SORT LOG</div>', unsafe_allow_html=True)
    log_html = ""
    for entry in list(s.log)[:20]:
        log_html += f'<div class="log-entry {entry["kind"]}"><span class="ts">{entry["ts"]}</span>{entry["msg"]}</div>'
    if not log_html:
        log_html = '<div class="log-entry"><span class="ts">--:--:--</span>No production telemetry.</div>'
    st.markdown(log_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN SYSTEM DASHBOARD VIEW
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:4px;padding-bottom:10px;
            border-bottom:1px solid #1e2530;'>
    <span style='font-family:Barlow Condensed,sans-serif;font-size:28px;font-weight:800;
                 letter-spacing:0.08em;color:#e8f0ff;'>FQA INSPECTION DESK</span>
    <span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#334055;
                 letter-spacing:0.15em;'>REAL-TIME COMPUTER VISION FRAME PIPELINE</span>
</div>
""", unsafe_allow_html=True)

# Process active ticks
process_frame()

col_prev, col_pred = st.columns(2, gap="medium")

# ── Raw Camera Feed Display ──
with col_prev:
    st.markdown('<div class="panel-header"><span>◈</span>CAMERA FEED — RAW LIVE OVERVIEW</div>', unsafe_allow_html=True)
    prev_placeholder = st.empty()
    if s.raw_frame is not None:
        prev_placeholder.image(cv2.cvtColor(s.raw_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        prev_placeholder.markdown("""
        <div style='background:#0d1017;border:1px solid #1e2530;border-radius:4px;
                    height:350px;display:flex;align-items:center;justify-content:center;
                    font-family:Share Tech Mono,monospace;font-size:12px;color:#334055;
                    letter-spacing:0.1em;'>
            CAMERA PIPELINE DISCONNECTED
        </div>
        """, unsafe_allow_html=True)

# ── Prediction Inference Overlay Display ──
with col_pred:
    st.markdown('<div class="panel-header"><span>◈</span>AI DETECTION — INFERENCE PREDICTION OVERLAY</div>', unsafe_allow_html=True)
    pred_placeholder = st.empty()
    if s.pred_frame is not None:
        pred_placeholder.image(cv2.cvtColor(s.pred_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        pred_placeholder.markdown("""
        <div style='background:#0d1017;border:1px solid #1e2530;border-radius:4px;
                    height:350px;display:flex;align-items:center;justify-content:center;
                    font-family:Share Tech Mono,monospace;font-size:12px;color:#334055;
                    letter-spacing:0.1em;'>
            INFERENCE CORES IDLE
        </div>
        """, unsafe_allow_html=True)

# ── Visual Color Legend ──
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
leg1, leg2, leg3 = st.columns(3)
legends = [
    ("#00ff99", "NON-ROTTEN / ACCEPTED FRUIT", "Any fruit not classified as rotten, spoiled, or defective."),
    ("#ff2244", "DEFECT / REJECTED OBJECT", "Items flagged as rotten, spoiled, or damaged."),
    ("#ffcc00", "UNKNOWN METRIC STATE",    "Items detected outside basic classification models."),
]
for col, (clr, title, desc) in zip([leg1, leg2, leg3], legends):
    with col:
        st.markdown(f"""
        <div style='border-left:3px solid {clr};padding:6px 10px;background:#0d1017;
                    border-radius:0 4px 4px 0;'>
            <div style='font-family:Share Tech Mono,monospace;font-size:10px;
                        color:{clr};letter-spacing:0.1em;'>{title}</div>
            <div style='font-size:11px;color:#445060;margin-top:2px;
                        font-family:Barlow Condensed,sans-serif;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Smooth Refresh Context ──
if s.running:
    time.sleep(0.01)
    st.rerun()