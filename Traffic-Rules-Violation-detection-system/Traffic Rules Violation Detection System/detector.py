# detector.py
from ultralytics import YOLO
from config import Config

class YOLODetector:
    """Modern object detector using YOLOv8."""

    def __init__(self):
        self.model = YOLO(Config.YOLO_MODEL)
        self.model.fuse()  # optimize inference

    def detect(self, frame):
        """Returns list of detections (bbox, label, confidence)."""
        results = self.model.predict(source=frame, conf=Config.TRACK_CONFIDENCE, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if label in ["car", "motorcycle", "bus", "truck"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "confidence": conf
                    })
        return detections
