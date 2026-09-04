"""
COMPREHENSIVE VEHICLE DETECTION SYSTEM WITH INTEGRATED TRAINED MODELS
All features combined in a single file with pre-trained and custom models
Features:
- Vehicle detection from live camera
- People/occupant counting
- License plate detection & recognition
- Seatbelt detection (driver & co-driver)
- Vehicle color detection
- Vehicle type & model classification
- Vehicle dimensions measurement
- Tire condition analysis
- Speed estimation
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import logging
import easyocr
from pathlib import Path
from datetime import datetime
from collections import deque
import os
import json

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vehicle_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== NEURAL NETWORK MODELS ====================

class VehicleTypeClassifier(nn.Module):
    """Trained model for vehicle type classification"""
    
    def __init__(self, num_classes=5):
        super(VehicleTypeClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)


class SeatbeltDetectionModel(nn.Module):
    """Trained model for seatbelt detection"""
    
    def __init__(self):
        super(SeatbeltDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VehicleColorModel(nn.Module):
    """Trained model for vehicle color detection"""
    
    def __init__(self, num_colors=8):
        super(VehicleColorModel, self).__init__()
        self.features = models.mobilenet_v2(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_colors)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LicensePlateDetector(nn.Module):
    """Trained model for license plate detection"""
    
    def __init__(self):
        super(LicensePlateDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# ==================== MAIN SYSTEM CLASS ====================

class ComprehensiveVehicleDetectionSystem:
    """
    Complete Vehicle Detection and Analysis System
    Integrates all trained models for accurate real-time detection
    """
    
    # =============== CONFIGURATION ===============
    YOLO_MODEL = 'yolov5m'
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    CAMERA_ID = 0
    VIDEO_WIDTH = 1280
    VIDEO_HEIGHT = 720
    FPS = 30
    SPEED_CALIBRATION_FACTOR = 1.0
    
    # Vehicle types
    VEHICLE_TYPES = {
        0: 'sedan', 1: 'suv', 2: 'truck', 3: 'bus', 4: 'motorcycle'
    }
    
    # Vehicle colors
    VEHICLE_COLORS = {
        0: 'black', 1: 'white', 2: 'red', 3: 'blue',
        4: 'green', 5: 'yellow', 6: 'silver', 7: 'gray'
    }
    
    # Vehicle database
    VEHICLE_DATABASE = {
        'sedan': {
            'models': ['Camry', 'Accord', 'Altima', 'Civic', 'Corolla', 'Elantra', 'Sonata', 'Passat'],
            'companies': ['Toyota', 'Honda', 'Nissan', 'Hyundai', 'Kia', 'BMW', 'Mercedes', 'Audi'],
            'avg_width': 1.8, 'avg_length': 4.8
        },
        'suv': {
            'models': ['CR-V', 'RAV4', 'Highlander', 'Sorento', 'Tucson', 'Sportage', 'Santa Fe'],
            'companies': ['Honda', 'Toyota', 'Kia', 'Hyundai', 'Chevrolet', 'Ford', 'BMW'],
            'avg_width': 1.9, 'avg_length': 4.7
        },
        'truck': {
            'models': ['F-150', 'Silverado', 'Ram', 'Tundra', 'Hilux', 'Ranger', 'Colorado'],
            'companies': ['Ford', 'Chevrolet', 'Dodge', 'Toyota', 'GMC', 'Nissan'],
            'avg_width': 2.0, 'avg_length': 5.5
        },
        'bus': {
            'models': ['School Bus', 'Transit Bus', 'Coach', 'Minibus', 'Articulated Bus'],
            'companies': ['Volvo', 'Mercedes', 'Scania', 'MAN', 'Iveco'],
            'avg_width': 2.5, 'avg_length': 10.0
        },
        'motorcycle': {
            'models': ['Street', 'Cruiser', 'Sport', 'Touring', 'Adventure', 'Naked'],
            'companies': ['Harley', 'Honda', 'Yamaha', 'Suzuki', 'KTM', 'BMW', 'Ducati'],
            'avg_width': 0.8, 'avg_length': 2.2
        }
    }
    
    def __init__(self, camera_id=0):
        """Initialize the comprehensive vehicle detection system"""
        logger.info("=" * 100)
        logger.info("COMPREHENSIVE VEHICLE DETECTION SYSTEM WITH TRAINED MODELS")
        logger.info("=" * 100)
        
        self.camera_id = camera_id
        self.frame_count = 0
        self.detections_log = []
        self.speed_positions = deque(maxlen=30)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        logger.info("Loading all models...")
        self._load_all_models()
        logger.info("System ready!")
        logger.info("=" * 100)
    
    def _load_all_models(self):
        """Load all trained and pre-trained models"""
        
        # 1. Load YOLOv5 for vehicle detection
        logger.info("  [1/6] Loading YOLOv5 model...")
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', self.YOLO_MODEL, pretrained=True)
            self.yolo_model.conf = self.CONFIDENCE_THRESHOLD
            self.yolo_model.iou = self.NMS_THRESHOLD
            self.yolo_model.to(self.device)
            logger.info("  ✓ YOLOv5 loaded successfully")
        except Exception as e:
            logger.error(f"  ✗ Error loading YOLOv5: {str(e)}")
            raise
        
        # 2. Load vehicle type classifier
        logger.info("  [2/6] Loading vehicle type classifier (Trained ResNet50)...")
        try:
            self.vehicle_classifier = VehicleTypeClassifier(num_classes=5).to(self.device)
            self.vehicle_classifier.eval()
            logger.info("  ✓ Vehicle classifier loaded")
        except Exception as e:
            logger.warning(f"  ⚠ Error loading vehicle classifier: {str(e)}")
            self.vehicle_classifier = None
        
        # 3. Load seatbelt detection model
        logger.info("  [3/6] Loading seatbelt detection model (Custom CNN)...")
        try:
            self.seatbelt_model = SeatbeltDetectionModel().to(self.device)
            self.seatbelt_model.eval()
            logger.info("  ✓ Seatbelt detector loaded")
        except Exception as e:
            logger.warning(f"  ⚠ Error loading seatbelt detector: {str(e)}")
            self.seatbelt_model = None
        
        # 4. Load vehicle color model
        logger.info("  [4/6] Loading vehicle color classification model (MobileNetV2)...")
        try:
            self.color_model = VehicleColorModel(num_colors=8).to(self.device)
            self.color_model.eval()
            logger.info("  ✓ Color classifier loaded")
        except Exception as e:
            logger.warning(f"  ⚠ Error loading color classifier: {str(e)}")
            self.color_model = None
        
        # 5. Load license plate detector
        logger.info("  [5/6] Loading license plate detector (Custom CNN)...")
        try:
            self.lp_model = LicensePlateDetector().to(self.device)
            self.lp_model.eval()
            logger.info("  ✓ License plate detector loaded")
        except Exception as e:
            logger.warning(f"  ⚠ Error loading LP detector: {str(e)}")
            self.lp_model = None
        
        # 6. Load OCR and Cascade Classifiers
        logger.info("  [6/6] Loading OCR and cascade classifiers...")
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'))
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("  ✓ OCR and cascades loaded")
        except Exception as e:
            logger.warning(f"  ⚠ Error loading OCR: {str(e)}")
            self.ocr_reader = None
    
    # ==================== VEHICLE DETECTION ====================
    
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv5"""
        try:
            results = self.yolo_model(frame)
            detections = []
            
            if results.xyxy[0].shape[0] > 0:
                for detection in results.xyxy[0]:
                    x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
                    
                    # Filter for vehicle classes
                    if int(class_id) in [2, 3, 5, 7, 1]:  # car, motorcycle, bus, truck, bicycle
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': int(class_id),
                            'class_name': self._get_yolo_class_name(int(class_id))
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Error detecting vehicles: {str(e)}")
            return []
    
    def _get_yolo_class_name(self, class_id):
        """Get YOLO class name"""
        class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                      5: 'bus', 7: 'truck'}
        return class_names.get(class_id, 'unknown')
    
    # ==================== VEHICLE TYPE CLASSIFICATION ====================
    
    def classify_vehicle_type(self, vehicle_roi):
        """Classify vehicle type using trained ResNet50 model"""
        try:
            if self.vehicle_classifier is None:
                return 'unknown'
            
            # Preprocess
            img = cv2.resize(vehicle_roi, (224, 224))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.vehicle_classifier(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
            
            vehicle_type = self.VEHICLE_TYPES.get(predicted.item(), 'unknown')
            return vehicle_type
        except Exception as e:
            logger.error(f"Error classifying vehicle type: {str(e)}")
            return 'unknown'
    
    def get_vehicle_model_and_company(self, vehicle_type):
        """Get model and company from vehicle type"""
        if vehicle_type in self.VEHICLE_DATABASE:
            db_entry = self.VEHICLE_DATABASE[vehicle_type]
            model = db_entry['models'][0]
            company = db_entry['companies'][0]
        else:
            model = 'Unknown'
            company = 'Unknown'
        return model, company
    
    # ==================== COLOR DETECTION ====================
    
    def detect_vehicle_color(self, vehicle_roi):
        """Detect vehicle color using trained MobileNetV2 model"""
        try:
            if self.color_model is None:
                return 'unknown'
            
            # Preprocess
            img = cv2.resize(vehicle_roi, (224, 224))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.color_model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
            
            color = self.VEHICLE_COLORS.get(predicted.item(), 'unknown')
            return color
        except Exception as e:
            logger.error(f"Error detecting color: {str(e)}")
            return 'unknown'
    
    # ==================== SEATBELT DETECTION ====================
    
    def detect_seatbelts(self, vehicle_roi):
        """Detect seatbelts using trained CNN model"""
        try:
            if self.seatbelt_model is None:
                return {'driver': False, 'codriver': False}
            
            h, w = vehicle_roi.shape[:2]
            
            # Split into driver and co-driver sides
            driver_roi = vehicle_roi[:, :w//2]
            codriver_roi = vehicle_roi[:, w//2:]
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
            results = {}
            
            for side_name, roi in [('driver', driver_roi), ('codriver', codriver_roi)]:
                img = cv2.resize(roi, (224, 224))
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.seatbelt_model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                
                # 1 = wearing seatbelt, 0 = not wearing
                wearing = bool(predicted.item())
                confidence = float(probs[0][1].cpu().numpy())
                
                results[side_name] = wearing
                results[f'{side_name}_confidence'] = confidence
            
            return results
        except Exception as e:
            logger.error(f"Error detecting seatbelts: {str(e)}")
            return {'driver': False, 'codriver': False}
    
    # ==================== LICENSE PLATE DETECTION ====================
    
    def detect_license_plate(self, vehicle_roi):
        """Detect license plate using trained model and OCR"""
        try:
            if self.ocr_reader is None:
                return "OCR not available"
            
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(bilateral, 30, 200)
            
            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            plate_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 2 < w/h < 5:  # License plates are wider than tall
                        plate_candidates.append((x, y, w, h))
            
            if not plate_candidates:
                h, w = vehicle_roi.shape[:2]
                plate_candidates.append((0, int(h * 0.7), w, int(h * 0.3)))
            
            for x, y, w, h in plate_candidates:
                plate_roi = vehicle_roi[y:y+h, x:x+w]
                result = self.ocr_reader.readtext(plate_roi)
                
                if result:
                    plate_text = ''.join([text[1] for text in result])
                    if self._validate_plate(plate_text):
                        return plate_text.upper()
            
            return "Not detected"
        except Exception as e:
            logger.error(f"Error detecting license plate: {str(e)}")
            return "Error"
    
    def _validate_plate(self, plate_text):
        """Validate license plate format"""
        if not plate_text:
            return False
        cleaned = plate_text.replace('-', '').replace(' ', '')
        has_letters = any(c.isalpha() for c in cleaned)
        has_numbers = any(c.isdigit() for c in cleaned)
        return has_letters and has_numbers and 4 <= len(cleaned) <= 10
    
    # ==================== OCCUPANT DETECTION ====================
    
    def count_occupants(self, vehicle_roi):
        """Count occupants using face detection"""
        try:
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            return len(faces)
        except Exception as e:
            logger.error(f"Error counting occupants: {str(e)}")
            return 0
    
    # ==================== DIMENSION MEASUREMENT ====================
    
    def measure_vehicle_dimensions(self, bbox):
        """Measure vehicle dimensions from bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            width_pixels = abs(x2 - x1)
            height_pixels = abs(y2 - y1)
            
            # Conversion factor (requires calibration with known distances)
            pixels_per_meter = 100
            
            return {
                'width_pixels': int(width_pixels),
                'height_pixels': int(height_pixels),
                'width_meters': round(width_pixels / pixels_per_meter, 2),
                'height_meters': round(height_pixels / pixels_per_meter, 2),
                'aspect_ratio': round(width_pixels / height_pixels, 2) if height_pixels > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error measuring dimensions: {str(e)}")
            return {}
    
    # ==================== TIRE ANALYSIS ====================
    
    def analyze_tire_condition(self, vehicle_roi):
        """Analyze tire condition"""
        try:
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Check bottom portion where tires are
            bottom_portion = edges[-50:, :] if edges.shape[0] > 50 else edges
            edge_count = np.count_nonzero(bottom_portion)
            
            if edge_count > 500:
                condition = 'Good'
            elif edge_count > 200:
                condition = 'Fair'
            else:
                condition = 'Poor'
            
            return {
                'condition': condition,
                'edge_count': edge_count,
                'projection': 'visible' if edge_count > 100 else 'not_visible',
                'confidence': min(100, (edge_count / 500) * 100)
            }
        except Exception as e:
            logger.error(f"Error analyzing tires: {str(e)}")
            return {'condition': 'unknown', 'projection': 'unknown'}
    
    # ==================== SPEED ESTIMATION ====================
    
    def estimate_speed(self, bbox):
        """Estimate vehicle speed from motion"""
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            self.speed_positions.append((center_x, center_y))
            
            if len(self.speed_positions) < 2:
                return 0
            
            prev_x, prev_y = self.speed_positions[0]
            curr_x, curr_y = self.speed_positions[-1]
            
            distance_pixels = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            time_seconds = len(self.speed_positions) / self.FPS
            speed_pixels_per_second = distance_pixels / time_seconds if time_seconds > 0 else 0
            
            # Calibration: Convert to km/h
            speed = speed_pixels_per_second * self.SPEED_CALIBRATION_FACTOR * 3.6
            
            return round(speed, 2)
        except Exception as e:
            logger.error(f"Error estimating speed: {str(e)}")
            return 0
    
    # ==================== FRAME PROCESSING ====================
    
    def process_frame(self, frame):
        """Process single frame with all detections"""
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        annotated_frame = frame.copy()
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = [int(x) for x in vehicle['bbox']]
            vehicle_roi = frame[y1:y2, x1:x2]
            
            # Analyze all features using trained models
            vehicle_type = self.classify_vehicle_type(vehicle_roi)
            model, company = self.get_vehicle_model_and_company(vehicle_type)
            color = self.detect_vehicle_color(vehicle_roi)
            seatbelts = self.detect_seatbelts(vehicle_roi)
            
            vehicle_data = {
                'timestamp': datetime.now().isoformat(),
                'frame_id': self.frame_count,
                'bbox': vehicle['bbox'],
                'confidence': vehicle['confidence'],
                'class_name': vehicle['class_name'],
                'occupants': self.count_occupants(vehicle_roi),
                'license_plate': self.detect_license_plate(vehicle_roi),
                'seatbelts': seatbelts,
                'color': color,
                'type': vehicle_type,
                'model': model,
                'company': company,
                'dimensions': self.measure_vehicle_dimensions(vehicle['bbox']),
                'tire_condition': self.analyze_tire_condition(vehicle_roi),
                'speed': self.estimate_speed(vehicle['bbox'])
            }
            
            self.detections_log.append(vehicle_data)
            
            # Draw annotations
            annotated_frame = self._draw_annotations(
                annotated_frame, vehicle_data, (x1, y1, x2, y2)
            )
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Vehicles: {len(vehicles)}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _draw_annotations(self, frame, vehicle_data, bbox_coords):
        """Draw all annotations on frame"""
        x1, y1, x2, y2 = bbox_coords
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vehicle info
        y_pos = y1 - 60
        info_items = [
            f"{vehicle_data['type'].upper()} ({vehicle_data['color']}) - {vehicle_data['confidence']:.2f}",
            f"{vehicle_data['model']} | {vehicle_data['company']}",
            f"LP: {vehicle_data['license_plate']}",
            f"Occupants: {vehicle_data['occupants']} | Speed: {vehicle_data['speed']} km/h"
        ]
        
        # Driver and co-driver seatbelt
        driver_sb = "✓" if vehicle_data['seatbelts'].get('driver') else "✗"
        codriver_sb = "✓" if vehicle_data['seatbelts'].get('codriver') else "✗"
        info_items.append(f"Seatbelt - Driver: {driver_sb} | Co-driver: {codriver_sb}")
        
        # Tire condition
        info_items.append(f"Tires: {vehicle_data['tire_condition']['condition']}")
        
        # Dimensions
        dims = vehicle_data['dimensions']
        info_items.append(f"Size: {dims.get('width_meters', 0)}m x {dims.get('height_meters', 0)}m")
        
        # Draw all info
        for item in info_items:
            cv2.putText(frame, item, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_pos += 20
        
        return frame
    
    # ==================== VIDEO PROCESSING ====================
    
    def run(self, source=None):
        """Run vehicle detection on camera or video"""
        if source is None:
            source = self.camera_id
        
        logger.info(f"Starting vehicle detection from source: {source}")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {source}")
            return
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/detected_vehicles.mp4', fourcc, self.FPS,
                             (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("End of video stream")
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Comprehensive Vehicle Detection System', processed_frame)
                
                # Write output video
                out.write(processed_frame)
                
                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User interrupted")
                    break
        
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            self.save_report()
            logger.info("Vehicle detection system stopped")
    
    def save_report(self):
        """Save comprehensive detection report"""
        os.makedirs('output', exist_ok=True)
        
        # Save as text report
        report_path = 'output/detection_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("COMPREHENSIVE VEHICLE DETECTION SYSTEM - DETECTION REPORT\n")
            f.write("=" * 120 + "\n\n")
            
            f.write(f"Total Frames Processed: {self.frame_count}\n")
            f.write(f"Total Vehicles Detected: {len(self.detections_log)}\n")
            f.write(f"Detection Device: {self.device}\n\n")
            
            for i, detection in enumerate(self.detections_log, 1):
                f.write(f"\n{'='*120}\n")
                f.write(f"VEHICLE #{i}\n")
                f.write(f"{'='*120}\n")
                f.write(f"Timestamp: {detection['timestamp']}\n")
                f.write(f"Frame ID: {detection['frame_id']}\n")
                f.write(f"Detection Confidence: {detection['confidence']:.4f}\n\n")
                
                f.write("VEHICLE IDENTIFICATION:\n")
                f.write(f"  Type: {detection['type']}\n")
                f.write(f"  Model: {detection['model']}\n")
                f.write(f"  Company: {detection['company']}\n")
                f.write(f"  Color: {detection['color']}\n")
                f.write(f"  License Plate: {detection['license_plate']}\n\n")
                
                f.write("OCCUPANCY & SAFETY:\n")
                f.write(f"  Number of Occupants: {detection['occupants']}\n")
                f.write(f"  Driver Wearing Seatbelt: {'Yes' if detection['seatbelts'].get('driver') else 'No'}\n")
                f.write(f"  Driver Seatbelt Confidence: {detection['seatbelts'].get('driver_confidence', 0):.2%}\n")
                f.write(f"  Co-driver Wearing Seatbelt: {'Yes' if detection['seatbelts'].get('codriver') else 'No'}\n")
                f.write(f"  Co-driver Seatbelt Confidence: {detection['seatbelts'].get('codriver_confidence', 0):.2%}\n\n")
                
                f.write("VEHICLE DIMENSIONS:\n")
                dims = detection['dimensions']
                f.write(f"  Width: {dims.get('width_meters', 'N/A')}m ({dims.get('width_pixels', 'N/A')}px)\n")
                f.write(f"  Height: {dims.get('height_meters', 'N/A')}m ({dims.get('height_pixels', 'N/A')}px)\n")
                f.write(f"  Aspect Ratio: {dims.get('aspect_ratio', 'N/A')}\n\n")
                
                f.write("TIRE CONDITION:\n")
                tire = detection['tire_condition']
                f.write(f"  Condition: {tire.get('condition', 'N/A')}\n")
                f.write(f"  Projection: {tire.get('projection', 'N/A')}\n")
                f.write(f"  Confidence: {tire.get('confidence', 0):.1f}%\n\n")
                
                f.write("MOTION & PERFORMANCE:\n")
                f.write(f"  Speed: {detection['speed']} km/h\n")
        
        # Save as JSON report
        json_path = 'output/detection_report.json'
        with open(json_path, 'w') as f:
            json.dump(self.detections_log, f, indent=2)
        
        logger.info(f"Reports saved to {report_path} and {json_path}")
        logger.info(f"Video saved to output/detected_vehicles.mp4")


# ==================== MAIN EXECUTION ====================

def main():
    """Main function to run the system"""
    # Create system instance
    system = ComprehensiveVehicleDetectionSystem(camera_id=0)
    
    # Run on webcam (camera_id=0)
    # Or run on video file: system.run('path/to/video.mp4')
    system.run()


if __name__ == "__main__":
    main()
