# tracker.py
import supervision as sv
from config import Config

class ObjectTracker:
    """Tracks objects using Supervision’s ByteTrack."""

    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.tracks = {}

    def update(self, detections, frame_shape):
        """Takes detections from YOLO and returns tracked objects with IDs."""
        boxes = [d["bbox"] for d in detections]
        confidences = [d["confidence"] for d in detections]
        class_names = [d["label"] for d in detections]

        xyxy = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_names)
        tracked = self.tracker.update_with_detections(xyxy)

        tracked_objs = []
        for i, (x1, y1, x2, y2) in enumerate(tracked.xyxy):
            tracked_objs.append({
                "id": int(tracked.tracker_id[i]),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "label": class_names[i],
                "confidence": confidences[i]
            })
        return tracked_objs
