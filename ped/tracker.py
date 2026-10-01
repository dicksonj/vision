"""
Minimal centroid tracker - no external tracking library. Matches each
frame's detections to existing tracks by nearest centroid, so we can:
  - count unique pedestrians entering the frame over a time window
  - flag a pedestrian who stays in frame without moving (loitering)
"""
import time


class Track:
    __slots__ = ("id", "cx", "cy", "box", "first_seen", "last_seen",
                 "anchor", "anchor_time", "stationary")

    def __init__(self, track_id, cx, cy, box, now):
        self.id = track_id
        self.cx, self.cy = cx, cy
        self.box = box
        self.first_seen = now
        self.last_seen = now
        self.anchor = (cx, cy)      # position last time it "moved"
        self.anchor_time = now
        self.stationary = False


class CentroidTracker:
    def __init__(self, max_distance_px: float = 80, disappear_after_s: float = 1.0,
                 move_threshold_px: float = 20, stationary_after_s: float = 2.0):
        self.max_distance_px = max_distance_px
        self.disappear_after_s = disappear_after_s
        self.move_threshold_px = move_threshold_px
        self.stationary_after_s = stationary_after_s
        self.tracks = {}
        self._next_id = 1
        self.entry_events = []  # timestamps of new pedestrians entering frame

    def update(self, boxes, now=None):
        """boxes: list of (x1, y1, x2, y2, score). Returns list of Track
        objects currently visible, each with .stationary set."""
        now = now or time.time()
        centroids = [((x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2)) for x1, y1, x2, y2, _ in boxes]

        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(centroids)))

        # Greedy nearest-centroid matching (fine for sparse pedestrian counts)
        pairs = []
        for tid in list(unmatched_tracks):
            t = self.tracks[tid]
            best_j, best_d = None, self.max_distance_px
            for j in unmatched_dets:
                cx, cy, _ = centroids[j]
                d = ((t.cx - cx) ** 2 + (t.cy - cy) ** 2) ** 0.5
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                pairs.append((tid, best_j))
        for tid, j in pairs:
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(j)
            cx, cy, box = centroids[j]
            t = self.tracks[tid]
            t.cx, t.cy, t.box, t.last_seen = cx, cy, box, now
            if ((cx - t.anchor[0]) ** 2 + (cy - t.anchor[1]) ** 2) ** 0.5 > self.move_threshold_px:
                t.anchor, t.anchor_time = (cx, cy), now  # it moved - reset the clock
            t.stationary = (now - t.anchor_time) >= self.stationary_after_s

        # New detections become new tracks (and count as "entered frame")
        for j in unmatched_dets:
            cx, cy, box = centroids[j]
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(tid, cx, cy, box, now)
            self.entry_events.append(now)

        # Drop tracks that vanished a while ago
        for tid in unmatched_tracks:
            if now - self.tracks[tid].last_seen > self.disappear_after_s:
                del self.tracks[tid]

        return list(self.tracks.values())

    def count_last(self, seconds: float, now=None) -> int:
        now = now or time.time()
        cutoff = now - seconds
        self.entry_events = [t for t in self.entry_events if t >= cutoff]
        return len(self.entry_events)
