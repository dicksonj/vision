"""
Lightweight pedestrian detector: TensorFlow Lite (LiteRT) SSD-MobileNet-v1
trained on COCO, keeping only the 'person' class. No PyTorch / Ultralytics.
Also contains the USB (V4L2) camera opener and optional fisheye undistortion.
"""
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


class PersonDetector:
    def __init__(self, model_path: str, labels_path: str, num_threads: int = 4):
        self.interp = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interp.allocate_tensors()

        inp = self.interp.get_input_details()[0]
        self.in_index = inp["index"]
        self.in_dtype = inp["dtype"]
        _, self.in_h, self.in_w, _ = inp["shape"]
        self.out = self.interp.get_output_details()

        with open(labels_path) as f:
            labels = [line.strip() for line in f]
        if labels and labels[0] == "???":   # this model's labelmap has a dummy first row
            labels = labels[1:]
        self.person_id = labels.index("person")

    def detect(self, frame_bgr: np.ndarray, min_score: float = 0.5):
        """Returns a list of (x1, y1, x2, y2, score) in frame pixel coordinates."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self.in_w, self.in_h)), cv2.COLOR_BGR2RGB)
        x = np.expand_dims(rgb, 0)
        if self.in_dtype == np.float32:
            x = (x.astype(np.float32) - 127.5) / 127.5

        self.interp.set_tensor(self.in_index, x)
        self.interp.invoke()

        # Standard TFLite SSD post-process outputs: boxes, classes, scores, count
        boxes = self.interp.get_tensor(self.out[0]["index"])[0]
        classes = self.interp.get_tensor(self.out[1]["index"])[0]
        scores = self.interp.get_tensor(self.out[2]["index"])[0]

        people = []
        for (ymin, xmin, ymax, xmax), cls, score in zip(boxes, classes, scores):
            if int(cls) == self.person_id and score >= min_score:
                x1, x2 = int(np.clip(xmin, 0, 1) * w), int(np.clip(xmax, 0, 1) * w)
                y1, y2 = int(np.clip(ymin, 0, 1) * h), int(np.clip(ymax, 0, 1) * h)
                people.append((x1, y1, x2, y2, float(score)))
        return people


def open_camera(device, width=1280, height=720, fps=30):
    """Open a USB UVC camera through V4L2, requesting MJPG (needed for high
    resolution at full frame rate over USB)."""
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def build_undistort_maps(calib_path: str, size, balance: float = 0.5):
    """Optional fisheye correction. calib_path is an .npz with K (3x3) and D (4x1)
    from cv2.fisheye.calibrate at the SAME resolution as the capture size (w, h)."""
    data = np.load(calib_path)
    K, D = data["K"], data["D"]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=balance)
    return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
