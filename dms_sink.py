import cv2
import mediapipe as mp
import time
import threading
import numpy as np
import streamlit as st
from collections import deque
from datetime import datetime

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
Gst.init(None)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DMS — Driver Monitor",
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
.metric-sub {
    font-size: 11px;
    color: #445060;
    margin-top: 4px;
    font-family: 'Share Tech Mono', monospace;
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
.status-warning    { background: #2a1800; color: #ffcc00; border: 1px solid #ffcc0033; }
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

/* ── slider / toggle ── */
.stSlider > div { color: #556070 !important; }

/* ── image frame ── */
[data-testid="stImage"] img {
    border: 1px solid #1e2530;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these)
# ─────────────────────────────────────────────
CAMERA_IP                 = "192.168.4.100"
CAMERA_WIDTH              = 3072
CAMERA_HEIGHT             = 2048
INFERENCE_WIDTH           = 640
INFERENCE_HEIGHT          = 426
DISPLAY_SIZE              = (640, 426)
DISTRACTION_THRESHOLD_SEC = 1.0
INFERENCE_SKIP_FRAMES     = 1


# ─────────────────────────────────────────────
#  SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────
def _init_state():
    defaults = dict(
        running=False,
        grabber=None,
        face_mesh=None,
        cap=None,
        # counters
        focused_count=0,
        distracted_count=0,
        warning_count=0,
        total_frames=0,
        # current state
        is_distracted=False,
        distracted_start_time=None,
        status="OFFLINE",
        elapsed_distracted=0.0,
        # frames
        raw_frame=None,
        pred_frame=None,
        # log (newest first, max 50)
        log=deque(maxlen=50),
        last_nose=(0.5, 0.5),
        frame_count=0,
        start_error=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
#  FRAME GRABBER THREAD (native GStreamer, no cv2.CAP_GSTREAMER)
# ─────────────────────────────────────────────
class GstFrameGrabber(threading.Thread):
    """
    Runs a GStreamer pipeline itself via PyGObject and pulls frames from
    a named appsink, converting each sample straight to a numpy BGR
    array. This bypasses cv2.VideoCapture(..., cv2.CAP_GSTREAMER)
    entirely, so it works even when the installed OpenCV build has no
    GStreamer support compiled in.
    """
    def __init__(self, pipeline_str, width, height):
        super().__init__(daemon=True)
        self.width   = width
        self.height  = height
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.error   = None

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink  = self.pipeline.get_by_name("sink")
        if self.appsink is None:
            raise RuntimeError(
                "Pipeline has no element named 'sink' — add name=sink to the appsink."
            )
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self._on_new_sample)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def _on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            self.error = "GStreamer: end of stream"
            print(f"[GstFrameGrabber] {self.error}")
            self.stop()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.error = f"GStreamer error: {err} ({debug})"
            print(f"[GstFrameGrabber] {self.error}")
            self.stop()

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf   = sample.get_buffer()
        caps  = sample.get_caps().get_structure(0)
        w     = caps.get_value("width")
        h     = caps.get_value("height")
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
            frame = frame.reshape((h, w, 3)).copy()  # BGR, per pipeline caps
        finally:
            buf.unmap(mapinfo)
        with self.lock:
            self.frame = frame
        return Gst.FlowReturn.OK

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except Exception as e:
            self.error = f"GLib main loop error: {e}"
            print(f"[GstFrameGrabber] {self.error}")

    def get_latest(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            print(f"[GstFrameGrabber] error setting state NULL: {e}")
        if self.loop.is_running():
            self.loop.quit()


# ─────────────────────────────────────────────
#  START / STOP CAMERA
# ─────────────────────────────────────────────
def start_camera():
    pipeline_str = (
        f"aravissrc camera-name={CAMERA_IP} ! "
        f"video/x-raw,format=GRAY8,width={CAMERA_WIDTH},height={CAMERA_HEIGHT} ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "appsink name=sink drop=true max-buffers=1 sync=false emit-signals=true"
    )

    try:
        grabber = GstFrameGrabber(pipeline_str, CAMERA_WIDTH, CAMERA_HEIGHT)
    except Exception as e:
        st.session_state.start_error = f"Failed to build GStreamer pipeline: {e}"
        return

    grabber.start()

    # give the pipeline a moment to reach PLAYING / report an error
    time.sleep(0.5)
    if grabber.error is not None:
        st.session_state.start_error = grabber.error
        grabber.stop()
        return

    st.session_state.start_error = None

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    st.session_state.cap        = None   # unused with the GStreamer grabber, kept for stop_camera()
    st.session_state.grabber    = grabber
    st.session_state.face_mesh  = face_mesh
    st.session_state.running    = True
    st.session_state.status     = "FOCUSED"
    _log("Camera stream started", kind="ok")


def stop_camera():
    if st.session_state.grabber:
        st.session_state.grabber.stop()
    if st.session_state.face_mesh:
        st.session_state.face_mesh.close()
    st.session_state.running    = False
    st.session_state.status     = "OFFLINE"
    st.session_state.grabber    = None
    st.session_state.cap        = None
    st.session_state.face_mesh  = None
    _log("Camera stream stopped", kind="warn")


def reset_counters():
    st.session_state.focused_count    = 0
    st.session_state.distracted_count = 0
    st.session_state.warning_count    = 0
    st.session_state.total_frames     = 0
    st.session_state.log.clear()
    _log("Counters reset", kind="ok")


# ─────────────────────────────────────────────
#  PROCESS ONE FRAME
# ─────────────────────────────────────────────
def process_frame():
    s = st.session_state
    if not s.running or s.grabber is None:
        return

    frame = s.grabber.get_latest()
    if frame is None:
        return

    s.total_frames += 1
    s.frame_count  += 1
    run_inference = (s.frame_count % (INFERENCE_SKIP_FRAMES + 1) == 0)

    current_distracted = False

    if run_inference:
        small     = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results   = s.face_mesh.process(rgb_small)

        if results.multi_face_landmarks:
            nose = results.multi_face_landmarks[0].landmark[1]
            s.last_nose = (nose.x, nose.y)
        else:
            s.last_nose = None

    if s.last_nose is not None:
        nx, ny = s.last_nose
        current_distracted = nx < 0.4 or nx > 0.6 or ny > 0.65

    # temporal logic
    now = time.time()
    if current_distracted:
        if s.distracted_start_time is None:
            s.distracted_start_time = now
            s.warning_count += 1
            _log("Warning threshold entered", kind="warn")
        s.elapsed_distracted = now - s.distracted_start_time
        if s.elapsed_distracted >= DISTRACTION_THRESHOLD_SEC:
            if not s.is_distracted:
                s.distracted_count += 1
                _log(f"DISTRACTION DETECTED ({s.distracted_count} total)", kind="crit")
            s.is_distracted = True
            s.status = "DISTRACTED"
        else:
            s.status = "WARNING"
    else:
        if s.is_distracted:
            s.focused_count += 1
            _log("Driver refocused", kind="ok")
        s.distracted_start_time = None
        s.is_distracted         = False
        s.elapsed_distracted    = 0.0
        s.status                = "FOCUSED"

    # ── raw preview frame (no overlays) ──
    s.raw_frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)

    # ── prediction frame (with overlays) ──
    pred = s.raw_frame.copy()
    color = (0, 0, 255) if s.is_distracted else (0, 255, 0) if s.status == "FOCUSED" else (0, 200, 255)

    # status banner
    cv2.rectangle(pred, (0, 0), (DISPLAY_SIZE[0], 44), (10, 12, 18), -1)
    cv2.putText(pred, f"STATUS: {s.status}", (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)

    # nose dot
    if s.last_nose is not None:
        nx, ny   = s.last_nose
        dot_x    = int(nx * DISPLAY_SIZE[0])
        dot_y    = int(ny * DISPLAY_SIZE[1])
        cv2.circle(pred, (dot_x, dot_y), 8, color, -1)
        cv2.circle(pred, (dot_x, dot_y), 12, color, 1)

    # attention zone overlay
    zone_x1 = int(0.4 * DISPLAY_SIZE[0])
    zone_x2 = int(0.6 * DISPLAY_SIZE[0])
    zone_y2 = int(0.65 * DISPLAY_SIZE[1])
    overlay  = pred.copy()
    cv2.rectangle(overlay, (zone_x1, 0), (zone_x2, zone_y2), (0, 80, 40), -1)
    cv2.addWeighted(overlay, 0.12, pred, 0.88, 0, pred)
    cv2.rectangle(pred, (zone_x1, 0), (zone_x2, zone_y2), (0, 180, 80), 1)

    # warning timer bar
    if s.distracted_start_time and not s.is_distracted:
        frac  = min(s.elapsed_distracted / DISTRACTION_THRESHOLD_SEC, 1.0)
        bw    = int(frac * DISPLAY_SIZE[0])
        cv2.rectangle(pred, (0, DISPLAY_SIZE[1] - 6), (bw, DISPLAY_SIZE[1]), (0, 200, 255), -1)

    # distraction flash border
    if s.is_distracted:
        cv2.rectangle(pred, (0, 0), (DISPLAY_SIZE[0] - 1, DISPLAY_SIZE[1] - 1), (0, 0, 255), 4)

    s.pred_frame = pred


def _log(msg: str, kind: str = ""):
    ts  = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.appendleft({"ts": ts, "msg": msg, "kind": kind})


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding:10px 0 20px'>
            <div style='font-family:Share Tech Mono,monospace;font-size:10px;
                        letter-spacing:0.2em;color:#445060;'>SYSTEM</div>
            <div style='font-family:Barlow Condensed,sans-serif;font-size:26px;
                        font-weight:800;letter-spacing:0.05em;color:#e8f0ff;line-height:1.1;'>
                DRIVER<br>MONITOR
            </div>
            <div style='font-family:Share Tech Mono,monospace;font-size:9px;
                        color:#334055;letter-spacing:0.15em;margin-top:4px;'>
                JETSON ORIN · MEDIAPIPE
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # status badge
    s = st.session_state
    badge_cls = {
        "FOCUSED":    "status-focused",
        "DISTRACTED": "status-distracted",
        "WARNING":    "status-warning",
        "OFFLINE":    "status-offline",
    }.get(s.status, "status-offline")
    st.markdown(f'<div class="status-badge {badge_cls}">{s.status}</div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # counters
    for label, val, cls in [
        ("DISTRACTION EVENTS", s.distracted_count, "danger"),
        ("WARNINGS TRIGGERED", s.warning_count,    "warn"),
        ("FOCUS RECOVERIES",   s.focused_count,    "ok"),
        ("FRAMES PROCESSED",   s.total_frames,     ""),
    ]:
        st.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶  START" if not s.running else "⏹  STOP"):
            if not s.running:
                start_camera()
            else:
                stop_camera()
            st.rerun()
    if s.start_error:
        st.error(f"❌  {s.start_error}")
    with col2:
        if st.button("↺  RESET"):
            reset_counters()
            st.rerun()

    st.markdown("---")

    # event log
    st.markdown('<div class="metric-label" style="margin-bottom:8px">EVENT LOG</div>',
                unsafe_allow_html=True)
    log_html = ""
    for entry in list(s.log)[:20]:
        log_html += (
            f'<div class="log-entry {entry["kind"]}">'
            f'<span class="ts">{entry["ts"]}</span>{entry["msg"]}</div>'
        )
    if not log_html:
        log_html = '<div class="log-entry"><span class="ts">--:--:--</span>No events yet.</div>'
    st.markdown(log_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:4px;padding-bottom:10px;
            border-bottom:1px solid #1e2530;'>
    <span style='font-family:Barlow Condensed,sans-serif;font-size:28px;font-weight:800;
                 letter-spacing:0.08em;color:#e8f0ff;'>DMS DASHBOARD</span>
    <span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#334055;
                 letter-spacing:0.15em;'>REAL-TIME HEAD POSE ANALYSIS</span>
</div>
""", unsafe_allow_html=True)

# process one tick
process_frame()

col_prev, col_pred = st.columns(2, gap="medium")

# ── Camera Preview ──
with col_prev:
    st.markdown(
        '<div class="panel-header"><span>◈</span>CAMERA PREVIEW — RAW FEED</div>',
        unsafe_allow_html=True
    )
    prev_placeholder = st.empty()
    if s.raw_frame is not None:
        prev_placeholder.image(
            cv2.cvtColor(s.raw_frame, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
    else:
        prev_placeholder.markdown("""
        <div style='background:#0d1017;border:1px solid #1e2530;border-radius:4px;
                    height:300px;display:flex;align-items:center;justify-content:center;
                    font-family:Share Tech Mono,monospace;font-size:12px;color:#334055;
                    letter-spacing:0.1em;'>
            NO SIGNAL
        </div>
        """, unsafe_allow_html=True)

# ── Prediction View ──
with col_pred:
    st.markdown(
        '<div class="panel-header"><span>◈</span>PREDICTION VIEW — INFERENCE OVERLAY</div>',
        unsafe_allow_html=True
    )
    pred_placeholder = st.empty()
    if s.pred_frame is not None:
        pred_placeholder.image(
            cv2.cvtColor(s.pred_frame, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )
    else:
        pred_placeholder.markdown("""
        <div style='background:#0d1017;border:1px solid #1e2530;border-radius:4px;
                    height:300px;display:flex;align-items:center;justify-content:center;
                    font-family:Share Tech Mono,monospace;font-size:12px;color:#334055;
                    letter-spacing:0.1em;'>
            NO SIGNAL
        </div>
        """, unsafe_allow_html=True)

# ── Legend ──
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
leg1, leg2, leg3, leg4 = st.columns(4)
legends = [
    ("#00ff99", "FOCUSED",         "Head within attention zone"),
    ("#ffcc00", "WARNING",         "Approaching threshold"),
    ("#ff2244", "DISTRACTED",      "Threshold exceeded"),
    ("#00e5ff", "ATTENTION ZONE",  "Green box = safe gaze region"),
]
for col, (clr, title, desc) in zip([leg1, leg2, leg3, leg4], legends):
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

# ── auto-refresh while running ──
if s.running:
    time.sleep(0.03)   # ~30 fps target
    st.rerun()