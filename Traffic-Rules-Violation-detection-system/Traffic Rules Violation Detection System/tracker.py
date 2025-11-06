# # tracker.py
import numpy as np
import supervision as sv
from config import Config

class ObjectTracker:
    """Tracks objects using Supervision’s ByteTrack."""

    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.tracks = {}

    def update(self, detections, frame_shape):
        """Takes detections from YOLO and returns tracked objects with IDs."""

        if not detections:
            # No detections — still update tracker with empty frame
            empty_dets = sv.Detections.empty()
            _ = self.tracker.update_with_detections(empty_dets)
            return []

        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidences = np.array([d["confidence"] for d in detections], dtype=np.float32)
        labels = [d["label"] for d in detections]

        dets = sv.Detections(xyxy=boxes, confidence=confidences)
        tracked = self.tracker.update_with_detections(dets)

        tracked_objs = []
        for i, (x1, y1, x2, y2) in enumerate(tracked.xyxy):
            obj_id = int(tracked.tracker_id[i])
            label = labels[i] if i < len(labels) else "unknown"
            conf = float(confidences[i]) if i < len(confidences) else 0.0

            tracked_objs.append({
                "id": obj_id,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "label": label,
                "confidence": conf
            })

        return tracked_objs
