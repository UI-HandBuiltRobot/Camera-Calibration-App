"""
Configuration file for Added-Mass-Lab GUI
Contains all constants and configuration parameters
"""

import cv2

# Video format configuration
DEFAULT_CODEC = 'MJPG'
DEFAULT_VIDEO_EXTENSION = '.avi'

# File format configurations
SUPPORTED_VIDEO_FORMATS = [
    ("AVI files", "*.avi"),
    ("MP4 files", "*.mp4"),
    ("All files", "*.*")
]

# Camera configuration
DEFAULT_CAMERA_BACKEND = cv2.CAP_DSHOW
CAMERA_SCAN_RANGE = 10  # Number of cameras to scan (0-9)

# Threading configuration
PREVIEW_UPDATE_INTERVAL = 0.033  # ~30 FPS
QUEUE_TIMEOUT = 0.1
MAX_FRAME_QUEUE_SIZE = 60
MAX_PREVIEW_QUEUE_SIZE = 5

# GUI configuration
MAIN_WINDOW_SIZE = "700x500"
CAMERA_SELECTION_WINDOW_SIZE = "800x800"
CALIBRATION_RECORDING_WINDOW_SIZE = "800x700"
CALIBRATION_PREVIEW_WINDOW_SIZE = "1300x700"
MEASUREMENT_RECORDING_WINDOW_SIZE = "800x700"

# Calibration configuration
DEFAULT_CHECKERBOARD_SIZE = (9, 6)
FALLBACK_CHECKERBOARD_SIZES = [(8, 6), (7, 5)]
MIN_CALIBRATION_FRAMES = 30
CALIBRATION_FRAME_ANALYSIS_COUNT = 40
MIN_PERSPECTIVE_SQUARE_SIZE = 10.0

# Visual cueing configuration for calibration recording
VISUAL_CUEING_ENABLED = True
CUEING_GRID_SIZE = (5, 5)  # 5x5 grid overlay
CUEING_POINT_THRESHOLDS = {
    'yellow': 400,  # Minimum points to turn sector yellow
    'green': 800    # Minimum points to turn sector green
}
CUEING_OVERLAY_ALPHA = 0.3  # Transparency (0.0 to 1.0)
CUEING_COLORS = {
    'red': (0, 0, 255),      # BGR format for OpenCV
    'yellow': (0, 255, 255), # BGR format for OpenCV
    'green': (0, 255, 0)     # BGR format for OpenCV
}

# Status colors
STATUS_COLORS = {
    'success': 'green',
    'error': 'red',
    'warning': 'orange',
    'info': 'blue'
}
