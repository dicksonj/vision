"""
Pedestrian monitoring demo - USB fisheye camera + TFLite SSD-MobileNet + Streamlit.

Run:  streamlit run ped_app.py
"""
import os
import threading
import time

import cv2
import streamlit as st

from detector import PersonDetector, build_undistort_maps, open_camera
from tracker import CentroidTracker

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "detect.tflite")
LABELS_PATH = os.path.join(HERE, "models", "labelmap.txt")

st.set_page_config(page_title="Pedestrian Monitor", layout="wide", page_icon="🚶")

# .streamlit/config.toml sets the base dark theme; this adds a bit more polish
# on top (card-style metrics, a red pulsing alert banner, tighter spacing).
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {
    background: #171b24;
    border: 1px solid #262b36;
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="stMetricValue"] {color: #e5e7eb;}
.stCaption, .stMarkdown p {color: #9ca3af;}
.alert-banner {
    background: #3b0d0d;
    border: 1px solid #ef4444;
    color: #fca5a5;
    border-radius: 10px;
    padding: 12px 16px;
    font-weight: 600;
    margin-bottom: 10px;
    animation: pulse 1.4s ease-in-out infinite;
}
.ok-banner {
    background: #0d2b17;
    border: 1px solid #22c55e;
    color: #86efac;
    border-radius: 10px;
    padding: 12px 16px;
    font-weight: 600;
    margin-bottom: 10px;
}
@keyframes pulse { 0%,100% {opacity: 1;} 50% {opacity: .55;} }
</style>
""", unsafe_allow_html=True)


class Grabber(threading.Thread):
    """Keeps reading the camera so the UI always shows the newest frame."""
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.frame, self.lock, self.running = cap, None, threading.Lock(), True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.02)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False


@st.cache_resource
def load_detector(threads: int):
    return PersonDetector(MODEL_PATH, LABELS_PATH, num_threads=threads)


for key, default in dict(running=False, grabber=None, cap=None, maps=None,
                         fps=0.0, last_t=None, error=None, tracker=None).items():
    st.session_state.setdefault(key, default)
s = st.session_state

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Camera")
    device = st.text_input("Device (index or path)", "0", help="e.g. 0 or /dev/video0")
    c1, c2, c3 = st.columns(3)
    width = c1.number_input("W", 320, 3840, 1280, 160)
    height = c2.number_input("H", 240, 2160, 720, 120)
    fps = c3.number_input("FPS", 5, 60, 30, 5)
    calib = st.text_input("Fisheye calib .npz (optional)", "")

    st.header("Detection")
    min_score = st.slider("Min confidence", 0.2, 0.95, 0.5, 0.05)
    threads = st.select_slider("CPU threads", options=[1, 2, 4, 6], value=4)

    st.header("Alerts & counting")
    count_window_min = st.slider("Counter window (minutes)", 1, 30, 10, 1)
    stationary_secs = st.slider("Flag as stationary after (seconds)", 1.0, 10.0, 2.0, 0.5)
    move_threshold = st.slider("Movement tolerance (pixels)", 5, 60, 20, 5,
                                help="A pedestrian must move more than this many "
                                     "pixels to reset the stationary timer - "
                                     "accounts for detector box jitter.")

    st.header("Display")
    display_fps = st.slider("Display refresh rate (fps)", 1, 15, 5, 1,
                             help="Lower = each frame (and its boxes/labels) stays "
                                  "on screen longer and is easier to read. The camera "
                                  "still grabs frames in the background at full speed; "
                                  "this only slows how often the UI redraws.")

    if st.button("⏹ Stop" if s.running else "▶ Start", use_container_width=True):
        if s.running:
            s.grabber.stop()
            s.cap.release()
            s.running, s.grabber, s.cap, s.maps = False, None, None, None
        else:
            cap = open_camera(device, int(width), int(height), int(fps))
            if not cap.isOpened():
                s.error = (f"Cannot open camera '{device}'. Check `v4l2-ctl --list-devices` "
                           "and that no other program is using it.")
            else:
                s.error = None
                s.cap = cap
                s.maps = None
                if calib and os.path.exists(calib):
                    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    s.maps = build_undistort_maps(calib, (aw, ah))
                s.grabber = Grabber(cap)
                s.grabber.start()
                s.tracker = CentroidTracker(stationary_after_s=stationary_secs,
                                             move_threshold_px=move_threshold)
                s.running, s.last_t = True, None
        st.rerun()

st.title("🚶 Pedestrian Monitor")
if s.error:
    st.error(s.error)

# Fail early with a clear message if the model files are missing
if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
    st.error(f"Model files not found in {os.path.join(HERE, 'models')}. See README steps.")
    st.stop()

detector = load_detector(threads)

# Static labels only - these don't change, so they stay outside the fragment.
raw_col, annotated_col = st.columns(2)
raw_col.caption("Raw feed")
annotated_col.caption("Detections")


@st.fragment(run_every=max(0.05, 1.0 / display_fps))
def render_video():
    """Owns and recreates its own layout every run. Streamlit fragments can
    only reserve a stable position for elements created INSIDE the fragment
    itself - reaching out to update a placeholder created in the outer
    script (even one written to once) isn't enough, hence this structure."""
    banner_slot = st.empty()
    img_col1, img_col2 = st.columns(2)
    raw_slot, annotated_slot = img_col1.empty(), img_col2.empty()
    m1, m2, m3 = st.columns(3)
    count_slot, window_slot, fps_slot = m1.empty(), m2.empty(), m3.empty()

    if not s.running:
        banner_slot.empty()
        annotated_slot.info("Stopped. Set the device in the sidebar and press ▶ Start.")
        return

    frame = s.grabber.latest()
    if frame is None:
        annotated_slot.info("Waiting for camera frames...")
        return

    if s.maps is not None:
        frame = cv2.remap(frame, s.maps[0], s.maps[1], cv2.INTER_LINEAR)

    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated = frame.copy()

    t0 = time.time()
    boxes = detector.detect(frame, min_score)
    infer_ms = (time.time() - t0) * 1000

    now = time.time()
    tracks = s.tracker.update(boxes, now)
    any_stationary = False

    for t in tracks:
        x1, y1, x2, y2 = t.box
        if t.stationary:
            any_stationary = True
            color = (0, 0, 255)  # red (BGR) - flagged
            label = f"#{t.id} STATIONARY {now - t.anchor_time:.0f}s"
        else:
            color = (0, 255, 0)  # green - normal
            label = f"#{t.id} person"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ly = max(th + 10, y1)
        cv2.rectangle(annotated, (x1, ly - th - 10), (x1 + tw + 10, ly), color, -1)
        text_color = (255, 255, 255) if t.stationary else (0, 0, 0)
        cv2.putText(annotated, label, (x1 + 5, ly - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    if s.last_t:
        s.fps = 0.9 * s.fps + 0.1 * (1.0 / max(now - s.last_t, 1e-3))
    s.last_t = now

    raw_slot.image(raw_rgb, use_container_width=True)
    annotated_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    count_slot.metric("Pedestrians in view", len(tracks))
    window_slot.metric(f"Entered in last {count_window_min} min",
                        s.tracker.count_last(count_window_min * 60, now))
    fps_slot.metric("Refresh FPS / inference", f"{s.fps:.1f} / {infer_ms:.0f} ms")

    if any_stationary:
        n = sum(1 for t in tracks if t.stationary)
        banner_slot.markdown(
            f'<div class="alert-banner">🚩 {n} pedestrian(s) stationary for '
            f'{stationary_secs:.0f}+ seconds</div>', unsafe_allow_html=True)
    else:
        banner_slot.markdown('<div class="ok-banner">✅ No stationary alerts</div>',
                              unsafe_allow_html=True)


render_video()
