# # main.py
# import cv2
# import time
# from config import Config
# from detector import YOLODetector
# from tracker import ObjectTracker
# from speed_estimator import SpeedEstimator
# from visualizer import Visualizer
# from logger_utils import log_event


# def main():
#     cap = cv2.VideoCapture(str(Config.VIDEO_PATH))
#     detector = YOLODetector()
#     tracker = ObjectTracker()
#     estimator = SpeedEstimator()
#     vis = Visualizer()

#     out = cv2.VideoWriter(str(Config.OUTPUT_PATH),
#                           cv2.VideoWriter_fourcc(*'MJPG'),
#                           Config.OUTPUT_FPS,
#                           (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

#         detections = detector.detect(frame)
#         tracked_objects = tracker.update(detections, frame.shape)

#         for obj in tracked_objects:
#             obj_id = obj["id"]
#             bbox = obj["bbox"]
#             label = obj["label"]
#             speed = estimator.estimate(obj_id, bbox)
#             vis.draw(frame, obj_id, bbox, label, speed)
#             if estimator.should_alert(obj_id, speed):
#                 log_event(obj_id, label, speed)
#             elif not Config.LOG_OVERSPEED_ONLY:
#                 log_event(obj_id, label, speed)


#         out.write(frame)
#         if Config.SHOW_DISPLAY:
#             cv2.imshow("Traffic Detection", frame)
#             if cv2.waitKey(1) == 27:  # ESC
#                 break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
# main.py
import cv2
import time
import logging
from config import Config
from detector import YOLODetector
from tracker import ObjectTracker
from speed_estimator import SpeedEstimator
from visualizer import Visualizer
from logger_utils import log_event


def main():
    # --- Video Initialization ---
    cap = cv2.VideoCapture(str(Config.VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"❌ Unable to open video file: {Config.VIDEO_PATH}")

    # --- Initialize modules ---
    detector = YOLODetector()
    tracker = ObjectTracker()
    estimator = SpeedEstimator()
    vis = Visualizer()

    # --- Detect and sync FPS ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1.0:  # Valid FPS detected
        estimator.nominal_fps = fps
        logging.info(f"[INFO] Video FPS detected: {fps:.2f}")
    else:
        estimator.nominal_fps = Config.OUTPUT_FPS or 30.0
        logging.warning(f"[WARN] FPS not found; defaulting to {estimator.nominal_fps:.2f}")

    # --- Prepare output writer ---
    out = cv2.VideoWriter(
        str(Config.OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*'MJPG'),
        estimator.nominal_fps,
        (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
    )

    logging.info("[INFO] Starting traffic detection... Press ESC to exit.")

    # --- Frame-by-frame processing loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("[INFO] End of video reached.")
            break

        # Resize large 4K frames to manageable size
        frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame.shape)

        # --- For each tracked object, estimate and visualize speed ---
        for obj in tracked_objects:
            obj_id = obj["id"]
            bbox = obj["bbox"]
            label = obj["label"]

            # Estimate smoothed speed
            speed = estimator.estimate(obj_id, bbox)
            vis.draw(frame, obj_id, bbox, label, speed)

            # Log overspeed or regular updates
            if estimator.should_alert(obj_id, speed):
                log_event(obj_id, label, speed)
            elif not Config.LOG_OVERSPEED_ONLY:
                log_event(obj_id, label, speed)

        # --- Output and display ---
        out.write(frame)
        if Config.SHOW_DISPLAY:
            cv2.imshow("Traffic Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                logging.info("[INFO] User exited manually.")
                break

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("[INFO] All resources released. Exiting cleanly.")


if __name__ == "__main__":
    main()
