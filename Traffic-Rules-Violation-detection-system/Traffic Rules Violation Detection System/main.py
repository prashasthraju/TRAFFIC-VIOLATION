# main.py
import cv2
import time
from config import Config
from detector import YOLODetector
from tracker import ObjectTracker
from speed_estimator import SpeedEstimator
from visualizer import Visualizer
from logger_utils import log_event

def main():
    cap = cv2.VideoCapture(str(Config.VIDEO_PATH))
    detector = YOLODetector()
    tracker = ObjectTracker()
    estimator = SpeedEstimator()
    vis = Visualizer()

    out = cv2.VideoWriter(str(Config.OUTPUT_PATH),
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          Config.OUTPUT_FPS,
                          (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame.shape)

        for obj in tracked_objects:
            obj_id = obj["id"]
            bbox = obj["bbox"]
            label = obj["label"]
            speed = estimator.estimate(obj_id, bbox)
            vis.draw(frame, obj_id, bbox, label, speed)
            log_event(obj_id, label, speed)

        out.write(frame)
        if Config.SHOW_DISPLAY:
            cv2.imshow("Traffic Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
