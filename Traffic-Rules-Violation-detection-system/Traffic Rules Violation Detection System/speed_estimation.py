# speed_estimator.py
import math
import time
from config import Config

class SpeedEstimator:
    """Computes object speed in km/h from tracked positions."""

    def __init__(self):
        self.previous_positions = {}
        self.previous_times = {}

    def estimate(self, obj_id, bbox):
        """Estimate instantaneous speed for tracked object."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        current_time = time.time()

        if obj_id in self.previous_positions:
            prev_cx, prev_cy = self.previous_positions[obj_id]
            dt = current_time - self.previous_times[obj_id]
            if dt > 0:
                pixel_distance = math.hypot(cx - prev_cx, cy - prev_cy)
                meters = pixel_distance / Config.PIXELS_PER_METER
                kmph = meters / dt * 3.6
            else:
                kmph = 0.0
        else:
            kmph = 0.0

        self.previous_positions[obj_id] = (cx, cy)
        self.previous_times[obj_id] = current_time
        return kmph
