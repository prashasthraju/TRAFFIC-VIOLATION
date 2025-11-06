# speed_estimator.py
import math
import time
import numpy as np
from collections import deque
from config import Config

class SpeedEstimator:
    """
    Robust, physically consistent speed estimator with temporal smoothing.
    """

    def __init__(self):
        self.history = {}  # id -> deque of (time, cx, cy)
        self.filtered_speed = {}  # id -> last smoothed speed
        self.last_alert_time = {}  # id -> debounce timer

        # Parameters
        self.window_size = 5        # frames used for smoothing
        self.alpha = 0.3            # exponential smoothing weight
        self.min_pixel_move = 5.0   # ignore <5px noise
        self.nominal_fps = 30.0     # assume stable FPS

    def estimate(self, obj_id, bbox):
        """Estimate speed (km/h) from centroid history with smoothing."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        now = time.time()

        # Initialize history
        if obj_id not in self.history:
            self.history[obj_id] = deque(maxlen=self.window_size)
            self.filtered_speed[obj_id] = 0.0
            self.last_alert_time[obj_id] = 0.0

        self.history[obj_id].append((now, cx, cy))

        # Not enough data
        if len(self.history[obj_id]) < 2:
            return 0.0

        # Compute displacement over N-frame window
        t0, x0, y0 = self.history[obj_id][0]
        t1, x1, y1 = self.history[obj_id][-1]

        pixel_dist = math.hypot(x1 - x0, y1 - y0)
        if pixel_dist < self.min_pixel_move:
            # Ignore micro-movements / jitter
            speed_kmph = 0.0
        else:
            dt = (t1 - t0)
            if dt <= 0:
                dt = 1.0 / self.nominal_fps
            meters = pixel_dist / Config.PIXELS_PER_METER
            speed_kmph = (meters / dt) * 3.6

        # Exponential smoothing
        prev = self.filtered_speed[obj_id]
        smooth_speed = self.alpha * speed_kmph + (1 - self.alpha) * prev
        self.filtered_speed[obj_id] = smooth_speed

        return smooth_speed

    def should_alert(self, obj_id, speed):
        """Returns True if object exceeds speed limit for >0.3s."""
        now = time.time()
        if speed > Config.SPEED_LIMIT_KMPH:
            last = self.last_alert_time[obj_id]
            if now - last > 0.3:  # debounce
                self.last_alert_time[obj_id] = now
                return True
        return False
