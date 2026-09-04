"""
Seatbelt Detector Module
Detects if driver and co-driver are wearing seatbelts
"""

import cv2
import numpy as np
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class SeatbeltDetector:
    """Detect seatbelt usage in vehicles"""
    
    def __init__(self):
        """Initialize seatbelt detector"""
        logger.info("Initializing Seatbelt Detector")
        self.seatbelt_cascade = self._load_cascade()
    
    def _load_cascade(self):
        """Load seatbelt cascade classifier"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"Error loading seatbelt cascade: {str(e)}")
            return None
    
    def detect_seatbelts(self, vehicle_roi):
        """
        Detect seatbelt usage for driver and co-driver
        
        Args:
            vehicle_roi: Region of interest (vehicle interior)
            
        Returns:
            Dictionary with driver and co-driver seatbelt status
        """
        try:
            seatbelt_info = {
                'driver': self._detect_seatbelt_on_person(vehicle_roi, position='driver'),
                'codriver': self._detect_seatbelt_on_person(vehicle_roi, position='codriver')
            }
            
            return seatbelt_info
        
        except Exception as e:
            logger.error(f"Error detecting seatbelts: {str(e)}")
            return {'driver': False, 'codriver': False}
    
    def _detect_seatbelt_on_person(self, vehicle_roi, position='driver'):
        """
        Detect seatbelt on a specific person
        
        Args:
            vehicle_roi: Vehicle region
            position: 'driver' or 'codriver'
            
        Returns:
            Boolean indicating seatbelt status
        """
        try:
            h, w = vehicle_roi.shape[:2]
            
            # Define regions for driver (left) and co-driver (right)
            if position == 'driver':
                person_roi = vehicle_roi[:, :w//2]
            else:
                person_roi = vehicle_roi[:, w//2:]
            
            # Detect faces (occupants)
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            ).detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return False
            
            # Check for seatbelt patterns
            # Seatbelts typically appear as dark diagonal straps
            seatbelt_detected = self._analyze_seatbelt_pattern(person_roi)
            
            return seatbelt_detected
        
        except Exception as e:
            logger.error(f"Error detecting seatbelt on {position}: {str(e)}")
            return False
    
    def _analyze_seatbelt_pattern(self, person_roi):
        """
        Analyze if seatbelt pattern is present
        
        Args:
            person_roi: Person region from vehicle
            
        Returns:
            Boolean indicating seatbelt presence
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for diagonal lines (characteristic of seatbelts)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            
            if lines is None:
                return False
            
            # Check for diagonal lines
            diagonal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate line angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Seatbelts are typically at 30-60 degrees
                if 30 < angle < 60 or 120 < angle < 150:
                    diagonal_lines += 1
            
            # If multiple diagonal lines found, likely a seatbelt
            return diagonal_lines > 2
        
        except Exception as e:
            logger.error(f"Error analyzing seatbelt pattern: {str(e)}")
            return False
    
    def get_seatbelt_status_text(self, seatbelt_info):
        """
        Get formatted seatbelt status text
        
        Args:
            seatbelt_info: Dictionary with seatbelt information
            
        Returns:
            Formatted status string
        """
        driver_status = "✓ Wearing" if seatbelt_info['driver'] else "✗ Not Wearing"
        codriver_status = "✓ Wearing" if seatbelt_info['codriver'] else "✗ Not Wearing"
        
        return f"Driver: {driver_status} | Co-driver: {codriver_status}"
    
    def log_seatbelt_violation(self, seatbelt_info, vehicle_info):
        """
        Log seatbelt violations
        
        Args:
            seatbelt_info: Seatbelt status
            vehicle_info: Vehicle information
            
        Returns:
            List of violations
        """
        violations = []
        
        if not seatbelt_info['driver']:
            violations.append("Driver not wearing seatbelt")
        
        if not seatbelt_info['codriver']:
            violations.append("Co-driver not wearing seatbelt")
        
        if violations:
            logger.warning(f"Seatbelt violations detected: {violations}")
            logger.warning(f"Vehicle: {vehicle_info.get('license_plate', 'Unknown')}")
        
        return violations
