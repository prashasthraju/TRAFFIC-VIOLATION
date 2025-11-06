# config.py
from pathlib import Path

class Config:
    # I/O paths
    VIDEO_PATH = Path("record.mkv")
    OUTPUT_PATH = Path("output.avi")

    # YOLO model path (auto-download if missing)
    YOLO_MODEL = "yolov8n.pt"  # lightweight, accurate

    # Processing
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    OUTPUT_FPS = 20

    # Calibration (adjust for your camera setup)
    PIXELS_PER_METER = 8.8
    SPEED_LIMIT_KMPH = 30

    # Tracking
    TRACK_CONFIDENCE = 0.25
    MAX_TRACK_LOST = 10

    # Display
    SHOW_DISPLAY = True
    FONT_SCALE = 0.6
    LINE_THICKNESS = 2

    # Logging
    LOG_OVERSPEED_ONLY = False
