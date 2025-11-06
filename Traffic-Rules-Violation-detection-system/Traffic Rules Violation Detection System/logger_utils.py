# logger_utils.py
import logging
from config import Config

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

def log_event(obj_id, label, speed):
    """Logs speed events."""
    if Config.LOG_OVERSPEED_ONLY and speed <= Config.SPEED_LIMIT_KMPH:
        return
    if speed > Config.SPEED_LIMIT_KMPH:
        logging.warning(f"🚨 OVERSPEED ALERT: {label.upper()} #{obj_id} @ {int(speed)} km/h")
    else:
        logging.info(f"{label.upper()} #{obj_id}: {int(speed)} km/h")
