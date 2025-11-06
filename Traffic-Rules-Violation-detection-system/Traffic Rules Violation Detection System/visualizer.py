# visualizer.py
import cv2
from config import Config

class Visualizer:
    """Draw bounding boxes, labels, and overspeed alerts."""

    def __init__(self):
        pass

    def draw(self, frame, obj_id, bbox, label, speed):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if speed <= Config.SPEED_LIMIT_KMPH else (0, 0, 255)
        label_text = f"{label.upper()} #{obj_id} | {int(speed)} km/h"
        if speed > Config.SPEED_LIMIT_KMPH:
            label_text += " ⚠️"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, Config.LINE_THICKNESS)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, color, 2)
