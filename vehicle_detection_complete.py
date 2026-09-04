"""
Comprehensive Vehicle Detection and Analysis System
All features integrated in a single file for ease of use
"""

import cv2
import numpy as np
import torch
import logging
import easyocr
from pathlib import Path
from datetime import datetime
from collections import deque
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vehicle_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveVehicleDetectionSystem:
    """
    Complete vehicle detection and analysis system with all features:
    - Object detection from live camera
    - People counting in vehicle
    - Vehicle number (license plate) detection
    - Seatbelt detection (driver and co-driver)
    - Vehicle color detection
    - Vehicle type classification
    - Vehicle model and company identification
    - Vehicle dimension measurement
    - Tire condition analysis
    - Speed estimation
    """
    
    # Configuration
    YOLO_MODEL = 'yolov5m'
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    CAMERA_ID = 0
    VIDEO_WIDTH = 1280
    VIDEO_HEIGHT = 720
    FPS = 30
    SPEED_CALIBRATION_FACTOR = 1.0
    
    # Vehicle color ranges (HSV)
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
    
    # Vehicle database
    VEHICLE_DATABASE = {
        'sedan': {
            'models': ['Camry', 'Accord', 'Altima', 'Civic', 'Corolla', 'Hyundai Elantra'],
            'companies': ['Toyota', 'Honda', 'Nissan', 'Hyundai', 'Kia'],
            'avg_width': 1.8,
            'avg_length': 4.8
        },
        'suv': {
            'models': ['CR-V', 'RAV4', 'Highlander', 'Sorento', 'Tucson'],
            'companies': ['Honda', 'Toyota', 'Kia', 'Hyundai'],
            'avg_width': 1.9,
            'avg_length': 4.7
        },
        'truck': {
            'models': ['F-150', 'Silverado', 'Ram', 'Tundra', 'Hilux'],
            'companies': ['Ford', 'Chevrolet', 'Dodge', 'Toyota'],
            'avg_width': 2.0,
            'avg_length': 5.5
        },
        'bus': {
            'models': ['School Bus', 'Transit Bus', 'Coach', 'Minibus'],
            'companies': ['Various'],
            'avg_width': 2.5,
            'avg_length': 10.0
        },
        'motorcycle': {
            'models': ['Street', 'Cruiser', 'Sport', 'Touring', 'Adventure'],
            'companies': ['Harley', 'Honda', 'Yamaha', 'Suzuki', 'KTM'],
            'avg_width': 0.8,
            'avg_length': 2.2
        }
    }
    
    def __init__(self, camera_id=0):
        """
        Initialize the comprehensive vehicle detection system
        
        Args:
            camera_id: Camera ID (default: 0 for webcam)
        """
        logger.info("=" * 80)
        logger.info("Initializing Comprehensive Vehicle Detection System")
        logger.info("=" * 80)
        
        self.camera_id = camera_id
        self.frame_count = 0
        self.detections_log = []
        self.previous_frame = None
        self.speed_positions = deque(maxlen=30)
        
        # Load models
        self._load_models()
        
        logger.info("System initialization complete!")
        logger.info("=" * 80)
    
    def _load_models(self):
        """Load all required models"""
        logger.info("Loading YOLOv5 model...")
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', self.YOLO_MODEL, pretrained=True)
            self.yolo_model.conf = self.CONFIDENCE_THRESHOLD
            self.yolo_model.iou = self.NMS_THRESHOLD
            
            if torch.cuda.is_available():
                self.yolo_model.to('cuda')
                logger.info("✓ YOLOv5 model loaded on GPU")
            else:
                logger.info("✓ YOLOv5 model loaded on CPU")
        except Exception as e:
            logger.error(f"✗ Error loading YOLOv5: {str(e)}")
            raise
        
        logger.info("Loading OCR model for license plate recognition...")
        try:
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("✓ OCR model loaded")
        except Exception as e:
            logger.warning(f"⚠ Warning loading OCR: {str(e)}")
            self.ocr_reader = None
        
        # Load cascade classifiers
        logger.info("Loading cascade classifiers...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.info("✓ Cascade classifiers loaded")
    
    # ==================== VEHICLE DETECTION ====================
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in frame using YOLOv5
        
        Args:
            frame: Input frame from camera
            
        Returns:
            List of vehicle detections
        """
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
                            'class_name': self._get_class_name(int(class_id))
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Error detecting vehicles: {str(e)}")
            return []
    
    def _get_class_name(self, class_id):
        """Get class name from YOLO class ID"""
        class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck'
        }
        return class_names.get(class_id, 'unknown')
    
    # ==================== PEOPLE COUNTING ====================
    
    def count_people_in_vehicle(self, vehicle_roi):
        """
        Count number of people in vehicle
        
        Args:
            vehicle_roi: Region of interest (vehicle area)
            
        Returns:
            Number of people detected
        """
        try:
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            occupant_count = len(faces)
            return occupant_count
        except Exception as e:
            logger.error(f"Error counting people: {str(e)}")
            return 0
    
    # ==================== LICENSE PLATE DETECTION ====================
    
    def detect_license_plate(self, vehicle_roi):
        """
        Detect and recognize license plate number
        
        Args:
            vehicle_roi: Vehicle region of interest
            
        Returns:
            Recognized license plate text
        """
        try:
            if self.ocr_reader is None:
                return "OCR not available"
            
            # Convert to grayscale
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edged = cv2.Canny(bilateral, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            plate_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 2 < w/h < 5:  # License plates are wider than tall
                        plate_candidates.append((x, y, w, h))
            
            # If no candidates, use bottom portion
            if not plate_candidates:
                h, w = vehicle_roi.shape[:2]
                plate_candidates.append((0, int(h * 0.7), w, int(h * 0.3)))
            
            # Perform OCR on candidates
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
    
    # ==================== SEATBELT DETECTION ====================
    
    def detect_seatbelts(self, vehicle_roi):
        """
        Detect seatbelt usage for driver and co-driver
        
        Args:
            vehicle_roi: Vehicle region of interest
            
        Returns:
            Dictionary with seatbelt status
        """
        try:
            h, w = vehicle_roi.shape[:2]
            
            seatbelt_info = {
                'driver': self._detect_seatbelt_on_side(vehicle_roi[:, :w//2]),
                'codriver': self._detect_seatbelt_on_side(vehicle_roi[:, w//2:])
            }
            
            return seatbelt_info
        except Exception as e:
            logger.error(f"Error detecting seatbelts: {str(e)}")
            return {'driver': False, 'codriver': False}
    
    def _detect_seatbelt_on_side(self, person_roi):
        """
        Detect seatbelt on specific side
        
        Args:
            person_roi: Person region
            
        Returns:
            Boolean indicating seatbelt presence
        """
        try:
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for diagonal lines (seatbelt characteristic)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            
            if lines is None:
                return False
            
            diagonal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 30 < angle < 60 or 120 < angle < 150:
                    diagonal_lines += 1
            
            return diagonal_lines > 2
        except Exception as e:
            logger.error(f"Error analyzing seatbelt: {str(e)}")
            return False
    
    # ==================== VEHICLE COLOR DETECTION ====================
    
    def detect_vehicle_color(self, vehicle_roi):
        """
        Detect vehicle color
        
        Args:
            vehicle_roi: Vehicle region of interest
            
        Returns:
            Detected color name
        """
        try:
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            
            max_color = None
            max_pixels = 0
            
            for color_name, (lower, upper) in self.COLOR_RANGES.items():
                lower = np.array(lower)
                upper = np.array(upper)
                
                mask = cv2.inRange(hsv, lower, upper)
                pixels = cv2.countNonZero(mask)
                
                if pixels > max_pixels:
                    max_pixels = pixels
                    max_color = color_name
            
            return max_color if max_color else 'unknown'
        except Exception as e:
            logger.error(f"Error detecting color: {str(e)}")
            return 'unknown'
    
    # ==================== VEHICLE TYPE CLASSIFICATION ====================
    
    def classify_vehicle_type(self, vehicle_roi, class_id):
        """
        Classify vehicle type, model, and company
        
        Args:
            vehicle_roi: Vehicle region of interest
            class_id: YOLO class ID
            
        Returns:
            Tuple of (vehicle_type, model, company)
        """
        try:
            class_to_type = {
                2: 'sedan', 3: 'motorcycle', 5: 'bus', 7: 'truck', 1: 'bicycle'
            }
            
            vehicle_type = class_to_type.get(int(class_id), 'unknown')
            
            if vehicle_type in self.VEHICLE_DATABASE:
                db_entry = self.VEHICLE_DATABASE[vehicle_type]
                model = db_entry['models'][0]
                company = db_entry['companies'][0]
            else:
                model = 'unknown'
                company = 'unknown'
            
            return vehicle_type, model, company
        except Exception as e:
            logger.error(f"Error classifying vehicle: {str(e)}")
            return 'unknown', 'unknown', 'unknown'
    
    # ==================== VEHICLE DIMENSION MEASUREMENT ====================
    
    def measure_vehicle_dimensions(self, bbox):
        """
        Measure vehicle dimensions from bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with dimensions
        """
        try:
            x1, y1, x2, y2 = bbox
            width_pixels = abs(x2 - x1)
            height_pixels = abs(y2 - y1)
            
            # Rough real-world estimation (requires camera calibration)
            # These are estimates based on typical vehicle proportions
            pixels_per_meter = 100  # This should be calibrated with known distances
            
            width_meters = width_pixels / pixels_per_meter
            height_meters = height_pixels / pixels_per_meter
            
            return {
                'width_pixels': width_pixels,
                'height_pixels': height_pixels,
                'width_meters': round(width_meters, 2),
                'height_meters': round(height_meters, 2),
                'aspect_ratio': round(width_pixels / height_pixels, 2) if height_pixels > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error measuring dimensions: {str(e)}")
            return {}
    
    # ==================== TIRE CONDITION ANALYSIS ====================
    
    def analyze_tire_condition(self, vehicle_roi):
        """
        Analyze tire condition and projection
        
        Args:
            vehicle_roi: Vehicle region of interest
            
        Returns:
            Tire condition assessment
        """
        try:
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Count edges in bottom portion (where tires are)
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
            return {'condition': 'unknown', 'edge_count': 0, 'projection': 'unknown'}
    
    # ==================== SPEED ESTIMATION ====================
    
    def estimate_speed(self, bbox):
        """
        Estimate vehicle speed
        
        Args:
            bbox: Current bounding box [x1, y1, x2, y2]
            
        Returns:
            Estimated speed in km/h
        """
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
            
            # Calibration factor (requires real-world calibration)
            speed = speed_pixels_per_second * self.SPEED_CALIBRATION_FACTOR * 3.6
            
            return round(speed, 2)
        except Exception as e:
            logger.error(f"Error estimating speed: {str(e)}")
            return 0
    
    # ==================== MAIN PROCESSING ====================
    
    def process_frame(self, frame):
        """
        Process single frame with all detections
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Annotated frame with all detections
        """
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        annotated_frame = frame.copy()
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = [int(x) for x in vehicle['bbox']]
            vehicle_roi = frame[y1:y2, x1:x2]
            
            # Analyze all features
            vehicle_data = {
                'timestamp': datetime.now(),
                'frame_id': self.frame_count,
                'bbox': vehicle['bbox'],
                'confidence': vehicle['confidence'],
                'class_name': vehicle['class_name'],
                'occupants': self.count_people_in_vehicle(vehicle_roi),
                'license_plate': self.detect_license_plate(vehicle_roi),
                'seatbelts': self.detect_seatbelts(vehicle_roi),
                'color': self.detect_vehicle_color(vehicle_roi),
                'type': self.classify_vehicle_type(vehicle_roi, vehicle['class_id'])[0],
                'model': self.classify_vehicle_type(vehicle_roi, vehicle['class_id'])[1],
                'company': self.classify_vehicle_type(vehicle_roi, vehicle['class_id'])[2],
                'dimensions': self.measure_vehicle_dimensions(vehicle['bbox']),
                'tire_condition': self.analyze_tire_condition(vehicle_roi),
                'speed': self.estimate_speed(vehicle['bbox'])
            }
            
            self.detections_log.append(vehicle_data)
            
            # Draw annotations on frame
            annotated_frame = self._draw_annotations(
                annotated_frame, vehicle_data, (x1, y1, x2, y2)
            )
        
        return annotated_frame
    
    def _draw_annotations(self, frame, vehicle_data, bbox_coords):
        """
        Draw all annotations on frame
        
        Args:
            frame: Input frame
            vehicle_data: Vehicle information
            bbox_coords: Bounding box coordinates
            
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = bbox_coords
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw vehicle type and color
        label1 = f"{vehicle_data['type'].upper()} ({vehicle_data['color']}) - {vehicle_data['confidence']:.2f}"
        cv2.putText(frame, label1, (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw model and company
        label2 = f"{vehicle_data['model']} - {vehicle_data['company']}"
        cv2.putText(frame, label2, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw license plate
        label3 = f"LP: {vehicle_data['license_plate']}"
        cv2.putText(frame, label3, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw occupants
        label4 = f"Occupants: {vehicle_data['occupants']}"
        cv2.putText(frame, label4, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw seatbelt status
        driver_sb = "✓" if vehicle_data['seatbelts']['driver'] else "✗"
        codriver_sb = "✓" if vehicle_data['seatbelts']['codriver'] else "✗"
        label5 = f"Seatbelt - Driver: {driver_sb} CoDriver: {codriver_sb}"
        cv2.putText(frame, label5, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw speed
        label6 = f"Speed: {vehicle_data['speed']} km/h"
        cv2.putText(frame, label6, (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 2)
        
        # Draw tire condition
        label7 = f"Tires: {vehicle_data['tire_condition']['condition']}"
        cv2.putText(frame, label7, (x1, y2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
        
        # Draw dimensions
        dims = vehicle_data['dimensions']
        label8 = f"Size: {dims.get('width_meters', 0)}m x {dims.get('height_meters', 0)}m"
        cv2.putText(frame, label8, (x1, y2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 0), 2)
        
        return frame
    
    def run(self, source=None):
        """
        Run the vehicle detection system
        
        Args:
            source: Camera ID or video file path
        """
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
                
                # Display frame
                cv2.imshow('Vehicle Detection System', processed_frame)
                
                # Write to output video
                out.write(processed_frame)
                
                # Exit on 'q' key
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
        report_path = 'output/detection_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE VEHICLE DETECTION SYSTEM REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Total Frames Processed: {self.frame_count}\n")
            f.write(f"Total Vehicles Detected: {len(self.detections_log)}\n\n")
            
            for i, detection in enumerate(self.detections_log, 1):
                f.write(f"\n{'='*100}\n")
                f.write(f"VEHICLE #{i}\n")
                f.write(f"{'='*100}\n")
                f.write(f"Timestamp: {detection['timestamp']}\n")
                f.write(f"Frame ID: {detection['frame_id']}\n")
                f.write(f"Confidence: {detection['confidence']:.2f}\n\n")
                
                f.write("VEHICLE IDENTIFICATION:\n")
                f.write(f"  Type: {detection['type']}\n")
                f.write(f"  Model: {detection['model']}\n")
                f.write(f"  Company: {detection['company']}\n")
                f.write(f"  Color: {detection['color']}\n")
                f.write(f"  License Plate: {detection['license_plate']}\n\n")
                
                f.write("OCCUPANCY & SAFETY:\n")
                f.write(f"  Number of Occupants: {detection['occupants']}\n")
                f.write(f"  Driver Wearing Seatbelt: {'Yes' if detection['seatbelts']['driver'] else 'No'}\n")
                f.write(f"  Co-driver Wearing Seatbelt: {'Yes' if detection['seatbelts']['codriver'] else 'No'}\n\n")
                
                f.write("VEHICLE CHARACTERISTICS:\n")
                dims = detection['dimensions']
                f.write(f"  Width: {dims.get('width_meters', 'N/A')}m ({dims.get('width_pixels', 'N/A')}px)\n")
                f.write(f"  Height: {dims.get('height_meters', 'N/A')}m ({dims.get('height_pixels', 'N/A')}px)\n")
                f.write(f"  Aspect Ratio: {dims.get('aspect_ratio', 'N/A')}\n\n")
                
                f.write("TIRE CONDITION:\n")
                tire = detection['tire_condition']
                f.write(f"  Condition: {tire.get('condition', 'N/A')}\n")
                f.write(f"  Projection: {tire.get('projection', 'N/A')}\n")
                f.write(f"  Confidence: {tire.get('confidence', 'N/A'):.1f}%\n\n")
                
                f.write("MOTION & PERFORMANCE:\n")
                f.write(f"  Speed: {detection['speed']} km/h\n")
        
        logger.info(f"Report saved to {report_path}")


def main():
    """Main function to run the system"""
    # Create system instance
    system = ComprehensiveVehicleDetectionSystem(camera_id=0)
    
    # Run on webcam (camera_id=0)
    # Or run on video file: system.run('path/to/video.mp4')
    system.run()


if __name__ == "__main__":
    main()
