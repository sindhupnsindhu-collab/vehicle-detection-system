# Configuration file for Vehicle Detection System

# Model configurations
YOLO_MODEL = 'yolov5m'  # Options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Camera and video configurations
CAMERA_ID = 0  # Default webcam
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30

# Detection classes
VEHICLE_CLASSES = {
    'car': 2,
    'truck': 7,
    'bus': 5,
    'motorcycle': 3,
    'bicycle': 1,
    'scooter': 32
}

PERSON_CLASS = 0
SEATBELT_CLASS = 'seatbelt'

# License Plate Configuration
LP_CONFIDENCE = 0.5
LP_MODEL_PATH = 'models/license_plate_detector.pt'

# Vehicle Color Detection
COLOR_RANGES = {
    'black': ([0, 0, 0], [180, 255, 50]),
    'white': ([0, 0, 200], [180, 30, 255]),
    'red': ([0, 50, 50], [10, 255, 255]),
    'blue': ([100, 50, 50], [130, 255, 255]),
    'green': ([50, 50, 50], [90, 255, 255]),
    'yellow': ([20, 100, 100], [40, 255, 255]),
    'silver': ([0, 0, 100], [180, 50, 200]),
    'gray': ([0, 0, 50], [180, 50, 150])
}

# Speed estimation configuration
SPEED_CALIBRATION_FACTOR = 1.0  # Adjust based on camera height and angle
SPEED_UNIT = 'km/h'  # Options: km/h, mph

# Seatbelt detection confidence
SEATBELT_CONFIDENCE = 0.6

# Output configurations
OUTPUT_FPS = 20
SAVE_VIDEO = True
VIDEO_OUTPUT_PATH = 'output/detected_vehicles.mp4'
IMAGE_OUTPUT_PATH = 'output/detected_images'

# Logging
LOG_FILE = 'logs/vehicle_detection.log'
LOG_LEVEL = 'INFO'
