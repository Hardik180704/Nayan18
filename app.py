import cv2
import numpy as np
import time
import pyttsx3
import threading
import math
from collections import deque
from ultralytics import YOLO
import torch
import os
import sounddevice as sd
import soundfile as sf
import queue
import requests
import json
from pathlib import Path

class EnhancedNayan:
    def __init__(self):
        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        
        # Set up voice options
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Use a female voice
        
        # Load the YOLO model
        self.model = YOLO("yolov8n.pt")  # Using the smallest model for faster inference
        
        # Load MiDaS model for depth estimation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device)
        self.midas.eval()
        
        # MiDaS transform
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.small_transform
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
            
        # Get camera parameters
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Reference sizes for distance estimation (in cm)
        self.reference_objects = {
            "person": 170,  # Average height in cm
            "car": 150,     # Average height in cm
            "bottle": 20,   # Average height in cm
            "chair": 80,    # Average height in cm
            "dog": 50,      # Average height in cm
            "cat": 30       # Average height in cm
        }
        
        # Set detection parameters
        self.conf_threshold = 0.5  # Confidence threshold for detections
        self.last_speech_time = time.time()
        self.speech_cooldown = 3.0  # Time between speech announcements
        self.scene_description_cooldown = 15.0  # Time between scene descriptions
        self.last_scene_time = 0
        self.current_detections = []
        self.speech_queue = []
        self.speaking = False
        
        # Path planning parameters
        self.obstacle_memory = deque(maxlen=10)  # Remember recent obstacles
        self.safe_path = None
        
        # Depth map parameters
        self.depth_map = None
        self.depth_colormap = None
        self.depth_threshold = 2.0  # Meters - objects closer than this are considered immediate obstacles
        
        # OCR integration data
        self.ocr_active = False
        self.ocr_cooldown = 10.0  # Time between OCR scans
        self.last_ocr_time = 0
        
        # Navigation history
        self.location_history = deque(maxlen=100)  # Store recent locations for backtracking
        self.landmarks = {}  # Store named landmarks
        
        # Audio cues settings
        self.audio_queue = queue.Queue()
        self.audio_thread = threading.Thread(target=self.audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Proximity alert sounds
        self.sound_dir = "sounds"
        os.makedirs(self.sound_dir, exist_ok=True)
        self.download_sound_if_needed("proximity_alert", "https://soundbible.com/grab.php?id=2197&type=mp3")
        self.download_sound_if_needed("object_detected", "https://soundbible.com/grab.php?id=2158&type=mp3")
        
        # Interactive mode parameters
        self.interactive_mode = False
        self.voice_commands = {
            "describe": self.cmd_describe_scene,
            "identify": self.cmd_identify_objects,
            "read": self.cmd_read_text,
            "navigate": self.cmd_navigate,
            "remember": self.cmd_remember_location,
            "locate": self.cmd_locate_landmark,
            "help": self.cmd_help
        }
        
        # Start the speech thread
        self.speech_thread = threading.Thread(target=self.speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Hazard detection
        self.hazard_types = {
            "stairs": ["stairs", "staircase", "step"],
            "water": ["puddle", "pool", "water"],
            "traffic": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "sharp": ["knife", "scissors", "glass"]
        }
        
        # Emotion detection model
        self.emotion_model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.load_emotion_model()
        
        # INR currency detection
        self.inr_denominations = {
            '10': {'color_low': [20, 100, 100], 'color_high': [30, 255, 255], 'size_range': (120, 160)},
            '20': {'color_low': [40, 100, 100], 'color_high': [80, 255, 255], 'size_range': (130, 170)},
            '50': {'color_low': [90, 100, 100], 'color_high': [110, 255, 255], 'size_range': (140, 180)},
            '100': {'color_low': [0, 100, 100], 'color_high': [10, 255, 255], 'size_range': (150, 190)},
            '200': {'color_low': [110, 100, 100], 'color_high': [130, 255, 255], 'size_range': (160, 200)},
            '500': {'color_low': [140, 100, 100], 'color_high': [160, 255, 255], 'size_range': (170, 210)},
            '2000': {'color_low': [160, 100, 100], 'color_high': [180, 255, 255], 'size_range': (180, 220)}
        }
        self.last_currency_detection_time = 0
        self.currency_cooldown = 5.0
    
    def load_emotion_model(self):
        """Load the emotion detection model"""
        try:
            # Try to load a pre-trained emotion detection model
            emotion_model_path = os.path.join(os.path.dirname(__file__), "emotion_model.h5")
            
            # Check if model file exists
            if not os.path.exists(emotion_model_path):
                print("Emotion model not found. Emotion detection will be limited.")
                self.emotion_model = None
                return
                
            # Load the model (this would require Keras/TensorFlow)
            try:
                from tensorflow import keras
                from keras.models import load_model
                self.emotion_model = load_model(emotion_model_path)
                print("Emotion model loaded successfully")
            except ImportError:
                print("Keras/TensorFlow not available. Emotion detection will be limited.")
                self.emotion_model = None
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.emotion_model = None
    
    def detect_emotions(self, frame):
        """Detect emotions in faces in the frame"""
        if self.emotion_model is None:
            return []
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            emotions = []
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize to 48x48 for emotion model
                resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
                
                # Normalize and reshape for model input
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                
                # Predict emotion
                predictions = self.emotion_model.predict(reshaped)
                emotion_idx = np.argmax(predictions)
                emotion = self.emotion_labels[emotion_idx]
                confidence = predictions[0][emotion_idx]
                
                emotions.append({
                    'face_box': (x, y, w, h),
                    'emotion': emotion,
                    'confidence': float(confidence)
                })
                
            return emotions
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return []
    
    def detect_inr_currency(self, frame):
        """Detect and count INR currency notes in the frame"""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            detected_notes = []
            
            for denomination, params in self.inr_denominations.items():
                # Create mask for the denomination color
                lower = np.array(params['color_low'])
                upper = np.array(params['color_high'])
                mask = cv2.inRange(hsv, lower, upper)
                
                # Apply morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    # Filter by size
                    area = cv2.contourArea(cnt)
                    if params['size_range'][0] < area < params['size_range'][1]:
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # Check aspect ratio (currency notes are rectangular)
                        aspect_ratio = float(w)/h
                        if 1.5 < aspect_ratio < 3.5:
                            detected_notes.append({
                                'denomination': denomination,
                                'box': (x, y, x+w, y+h),
                                'area': area
                            })
            
            # Count notes by denomination
            currency_count = {}
            total_value = 0
            
            for note in detected_notes:
                denom = note['denomination']
                if denom in currency_count:
                    currency_count[denom] += 1
                else:
                    currency_count[denom] = 1
                total_value += int(denom)
            
            return detected_notes, currency_count, total_value
            
        except Exception as e:
            print(f"Error in currency detection: {e}")
            return [], {}, 0
    
    def download_sound_if_needed(self, name, url):
        """Download sound files if they don't exist"""
        file_path = os.path.join(self.sound_dir, f"{name}.mp3")
        if not os.path.exists(file_path):
            try:
                response = requests.get(url)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {name} sound file")
            except Exception as e:
                print(f"Failed to download sound: {e}")
    
    def speech_worker(self):
        """Background thread to handle speech without blocking the main thread"""
        while True:
            if self.speech_queue and not self.speaking:
                self.speaking = True
                announcement = self.speech_queue.pop(0)
                self.engine.say(announcement)
                self.engine.runAndWait()
                self.speaking = False
            time.sleep(0.1)
    
    def audio_worker(self):
        """Background thread to handle audio alerts"""
        while True:
            try:
                if not self.audio_queue.empty():
                    sound_file = self.audio_queue.get()
                    if os.path.exists(sound_file):
                        data, fs = sf.read(sound_file)
                        sd.play(data, fs)
                        sd.wait()
                time.sleep(0.1)
            except Exception as e:
                print(f"Audio error: {e}")
    
    def play_sound(self, sound_name):
        """Add sound to the audio queue"""
        sound_file = os.path.join(self.sound_dir, f"{sound_name}.mp3")
        self.audio_queue.put(sound_file)
    
    def announce(self, text, priority=False):
        """Add text to speech queue, with optional priority flag"""
        if priority:
            self.speech_queue.insert(0, text)
        else:
            self.speech_queue.append(text)
    
    def estimate_distance_with_midas(self, depth_map, box):
        """Estimate distance using MiDaS depth map"""
        x1, y1, x2, y2 = box
        # Extract the depth region for the object
        object_depth_region = depth_map[y1:y2, x1:x2]
        if object_depth_region.size == 0:
            return None
            
        # Use the median depth value for robustness
        median_depth = np.median(object_depth_region)
        
        # Convert relative depth to approximate meters
        # This is a simplification and would need calibration for accurate values
        depth_meters = median_depth * 10.0  # Scale factor needs calibration
        
        return depth_meters
    
    def estimate_distance(self, box, class_name, depth_map=None):
        """Estimate distance to object using either MiDaS or apparent size"""
        # If we have a depth map, use it for more accurate distance estimation
        if depth_map is not None:
            midas_distance = self.estimate_distance_with_midas(depth_map, box)
            if midas_distance is not None:
                return midas_distance
                
        # Fallback to reference size method
        if class_name not in self.reference_objects:
            return None
            
        # Extract box dimensions
        x1, y1, x2, y2 = box
        object_height_pixels = y2 - y1
        
        # Simple inverse proportion to estimate distance
        reference_height_cm = self.reference_objects[class_name]
        focal_length = 500  # Approximation
        
        distance_cm = (reference_height_cm * focal_length) / object_height_pixels
        return distance_cm / 100  # Convert to meters
    
    def get_direction(self, box):
        """Determine direction of object relative to center of frame"""
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) // 2
        
        # Calculate horizontal position relative to center
        rel_x = box_center_x - self.frame_center_x
        
        # Define direction based on position
        if abs(rel_x) < self.frame_width * 0.1:
            return "directly ahead"
        elif rel_x < -self.frame_width * 0.3:
            return "far to your left"
        elif rel_x < 0:
            return "to your left"
        elif rel_x > self.frame_width * 0.3:
            return "far to your right"
        else:
            return "to your right"
    
    def detect_hazards(self, detection_data, depth_map):
        """Identify potential hazards in the environment"""
        hazards = []
        
        # Check for known hazardous objects
        for obj in detection_data:
            class_name = obj['class']
            
            # Check if the object belongs to any hazard category
            for hazard_type, keywords in self.hazard_types.items():
                if any(keyword in class_name for keyword in keywords):
                    distance = obj['distance']
                    if distance and distance < 3.0:  # Only alert for close hazards
                        hazards.append({
                            'type': hazard_type,
                            'object': class_name,
                            'distance': distance,
                            'direction': obj['direction']
                        })
        
        # Use depth map to detect unmarked obstacles (like walls, furniture without labels)
        if depth_map is not None:
            # Look at the lower half of the depth map (ground obstacles)
            lower_half = depth_map[self.frame_height//2:, :]
            
            # Find regions that are very close
            close_mask = lower_half < self.depth_threshold
            
            if np.any(close_mask):
                # Identify the direction of the closest point
                close_points = np.where(close_mask)
                if len(close_points[0]) > 0:
                    # Find the closest point
                    closest_idx = np.argmin(lower_half[close_mask])
                    y, x = close_points[0][closest_idx], close_points[1][closest_idx]
                    
                    # Adjust y coordinate to account for the lower half
                    y += self.frame_height // 2
                    
                    # Calculate direction
                    if x < self.frame_width * 0.33:
                        direction = "to your left"
                    elif x > self.frame_width * 0.66:
                        direction = "to your right"
                    else:
                        direction = "directly ahead"
                    
                    # Add to hazards
                    hazards.append({
                        'type': 'unmarked_obstacle',
                        'object': 'obstacle',
                        'distance': lower_half[y-self.frame_height//2, x] * 10.0,  # Convert to meters
                        'direction': direction
                    })
        
        return hazards
    
    def plan_safe_path(self, obstacles, depth_map=None):
        """Find a safe path through obstacles using both object detection and depth information"""
        # If no obstacles detected and no depth map, path is clear
        if not obstacles and depth_map is None:
            return "Path is clear ahead"
            
        # Divide the frame into 7 vertical sections for more granular navigation
        section_width = self.frame_width // 7
        sections = [0] * 7  # Danger score for each section
        
        # Score sections based on detected obstacles
        for obj in obstacles:
            x1, y1, x2, y2 = obj['box']
            center_x = (x1 + x2) // 2
            section = min(6, center_x // section_width)
            
            # Weight by distance - closer objects are more dangerous
            distance = obj['distance'] if obj['distance'] else 5.0
            danger = 10.0 / (distance + 0.1)  # Avoid division by zero
            sections[section] += danger
        
        # Use depth information to enhance path planning
        if depth_map is not None:
            # Consider the bottom half of the frame (ground level obstacles)
            lower_region = depth_map[self.frame_height//2:, :]
            
            # Divide into sections matching our planning sections
            for i in range(7):
                start_x = i * section_width
                end_x = (i + 1) * section_width
                section_depth = lower_region[:, start_x:end_x]
                
                # Calculate danger based on proximity of objects
                if section_depth.size > 0:
                    # Find closest points in this section
                    min_depth = np.min(section_depth)
                    
                    # Add danger score based on depth
                    depth_danger = 5.0 / (min_depth * 10.0 + 0.1)  # Convert to meters
                    sections[i] += depth_danger
        
        # Find section with lowest danger score
        min_danger = min(sections)
        best_sections = [i for i, danger in enumerate(sections) if danger == min_danger]
        
        # Choose the most central safe section
        preferred_section = min(best_sections, key=lambda x: abs(x - 3))
        
        # Generate navigation guidance
        if min_danger < 1.0:  # Very safe path
            if preferred_section == 3:
                return "Path is clear directly ahead"
            elif preferred_section < 3:
                shift = "slightly" if preferred_section >= 2 else "more"
                return f"Move {shift} to the left for clearest path"
            else:
                shift = "slightly" if preferred_section <= 4 else "more"
                return f"Move {shift} to the right for clearest path"
        elif min_danger < 3.0:  # Somewhat safe path
            if preferred_section == 3:
                return "Proceed with caution straight ahead"
            elif preferred_section < 3:
                shift = "slightly" if preferred_section >= 2 else "more"
                return f"Move {shift} to the left and proceed with caution"
            else:
                shift = "slightly" if preferred_section <= 4 else "more"
                return f"Move {shift} to the right and proceed with caution"
        else:  # No safe path
            return "Caution! All paths have obstacles. Consider stopping or turning around"
    
    def perform_ocr(self, frame):
        """Extract text from the frame using Tesseract OCR"""
        try:
            import pytesseract
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(gray)
            
            # Clean and filter text
            if text.strip():
                filtered_text = ' '.join([line for line in text.split('\n') if len(line.strip()) > 3])
                if filtered_text:
                    return filtered_text
            return None
        except ImportError:
            self.announce("OCR feature requires pytesseract to be installed.")
            return None
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def generate_scene_description(self, detection_data, depth_data=None):
        """Generate a comprehensive description of the scene"""
        if not detection_data and depth_data is None:
            return "I don't see any objects clearly in the scene"
            
        # Count object types
        object_counts = {}
        for obj in detection_data:
            class_name = obj['class']
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1
        
        # Categorize the scene
        indoor_indicators = ["chair", "couch", "bed", "tv", "dining table", "refrigerator", "oven", "sink", "toilet", "desk"]
        outdoor_indicators = ["car", "bicycle", "traffic light", "tree", "bench", "bus"]
        
        indoor_score = sum(object_counts.get(item, 0) for item in indoor_indicators)
        outdoor_score = sum(object_counts.get(item, 0) for item in outdoor_indicators)
        
        scene_type = "indoors" if indoor_score > outdoor_score else "outdoors"
        
        # Count people and obstacles
        people_count = object_counts.get("person", 0)
        obstacle_count = sum(object_counts.values()) - people_count
        
        # Generate description
        description = f"You appear to be {scene_type}. "
        
        # Add depth context if available
        if depth_data is not None:
            # Calculate average depth in different regions
            top_half = depth_data[:self.frame_height//2, :]
            bottom_half = depth_data[self.frame_height//2:, :]
            
            avg_distance_ahead = np.median(depth_data) * 10.0  # Convert to meters
            
            if avg_distance_ahead < 1.5:
                description += "You're in a confined space. "
            elif avg_distance_ahead < 3.0:
                description += "You're in a medium-sized space. "
            else:
                description += "You're in an open area. "
            
            # Detect walls or large surfaces
            if np.std(top_half) < 0.1 and np.median(top_half) < 0.5:
                description += "There might be a wall close in front of you. "
            
            if np.std(bottom_half) < 0.1:
                if np.median(bottom_half) < 0.2:
                    description += "The floor appears very close, you might be looking down. "
        
        if people_count > 0:
            description += f"There {'is' if people_count == 1 else 'are'} {people_count} {'person' if people_count == 1 else 'people'} around you. "
        
        if obstacle_count > 0:
            description += f"I can see {obstacle_count} objects that could be obstacles. "
            
        # Add details about prominent objects
        prominent_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        if prominent_objects:
            objects_text = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" for name, count in prominent_objects])
            description += f"The main objects I detect are {objects_text}."
            
        return description
    
    def process_frame(self, frame):
        """Process a single frame with YOLO detection and MiDaS depth estimation"""
        # Run YOLO detection
        results = self.model(frame, conf=self.conf_threshold)
        
        # Process depth with MiDaS
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        with torch.no_grad():
            depth_prediction = self.midas(input_batch)
            depth_prediction = torch.nn.functional.interpolate(
                depth_prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = depth_prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        normalized_depth = normalized_depth.astype(np.uint8)
        self.depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
        
        # Store depth map for other functions to use
        self.depth_map = depth_map
        
        # Extract detections
        detection_data = []
        obstacles = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence
                confidence = float(box.conf)
                
                # Get class name
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Estimate distance using both methods
                distance = self.estimate_distance((x1, y1, x2, y2), class_name, depth_map)
                
                # Get direction
                direction = self.get_direction((x1, y1, x2, y2))
                
                # Add to detections
                obj_data = {
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2),
                    'distance': distance,
                    'direction': direction
                }
                detection_data.append(obj_data)
                
                # Add to obstacles list if relevant
                if class_name not in ["wall", "ceiling", "floor", "sky"] and y2 > self.frame_height * 0.5:
                    obstacles.append(obj_data)
                
                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display object info with distance
                label = f"{class_name}"
                if distance:
                    label += f" {distance:.1f}m"
                label += f" {direction}"
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update obstacle memory
        if obstacles:
            self.obstacle_memory.append(obstacles)
        
        # Check for hazards using both object detection and depth information
        hazards = self.detect_hazards(detection_data, depth_map)
        
        # Plan safe path considering both obstacles and depth map
        if self.obstacle_memory:
            all_obstacles = [item for sublist in self.obstacle_memory for item in sublist]
            self.safe_path = self.plan_safe_path(all_obstacles, depth_map)
            
            # Display path guidance on screen
            cv2.putText(frame, self.safe_path, (10, self.frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update current detections for use by other functions
        self.current_detections = detection_data
        
        # OCR processing if needed
        current_time = time.time()
        if self.ocr_active and current_time - self.last_ocr_time > self.ocr_cooldown:
            text = self.perform_ocr(frame)
            if text:
                self.announce(f"Text detected: {text}")
            self.last_ocr_time = current_time
        
        # Prepare speech announcement if needed
        if current_time - self.last_speech_time > self.speech_cooldown:
            announcements = []
            
            # Handle hazard warnings first (priority)
            if hazards:
                for hazard in hazards:
                    if hazard['type'] == 'unmarked_obstacle':
                        warning = f"Warning: Obstacle {hazard['direction']} about {hazard['distance']:.1f} meters away"
                    else:
                        warning = f"Caution: {hazard['object']} {hazard['direction']} about {hazard['distance']:.1f} meters away"
                    announcements.append(warning)
                    
                    # Play audio alert for very close hazards
                    if hazard['distance'] < 1.5:
                        self.play_sound("proximity_alert")
            
            # Add path guidance
            if self.safe_path:
                announcements.append(self.safe_path)
                
            # Object announcements
            # Sort objects by distance (closest first)
            sorted_objects = sorted([obj for obj in detection_data if obj['distance']], 
                                   key=lambda x: x['distance'])
            
            # Announce closest objects
            if sorted_objects:
                closest_objects = sorted_objects[:2]  # Limit to 2 closest objects
                
                obj_announcement = "I detect "
                obj_descriptions = []
                
                for obj in closest_objects:
                    desc = f"a {obj['class']} {obj['direction']}"
                    if obj['distance']:
                        desc += f", about {obj['distance']:.1f} meters away"
                    obj_descriptions.append(desc)
                
                obj_announcement += ", ".join(obj_descriptions)
                announcements.append(obj_announcement)
            
            # Special alerts for important objects
            if any(obj['class'] == "traffic light" for obj in detection_data):
                announcements.append("Traffic light detected")
                
            # Make the announcement
            if announcements:
                self.announce(". ".join(announcements))
                self.last_speech_time = current_time
        
        # Scene description (less frequent)
        if current_time - self.last_scene_time > self.scene_description_cooldown:
            scene_description = self.generate_scene_description(detection_data, depth_map)
            self.announce(scene_description)
            self.last_scene_time = current_time
        
        # Create a composite view with depth map
        composite_frame = np.zeros((self.frame_height, self.frame_width*2, 3), dtype=np.uint8)
        composite_frame[:, :self.frame_width] = frame
        if self.depth_colormap is not None:
            composite_frame[:, self.frame_width:] = self.depth_colormap
            
        # Add a separator line
        cv2.line(composite_frame, (self.frame_width, 0), (self.frame_width, self.frame_height), (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(composite_frame, "Object Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(composite_frame, "Depth Map", (self.frame_width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return composite_frame
    
    # Voice command handlers
    def cmd_describe_scene(self):
        """Voice command to describe the current scene"""
        scene_description = self.generate_scene_description(self.current_detections, self.depth_map)
        self.announce(scene_description, priority=True)
        return scene_description
    
    def cmd_identify_objects(self):
        """Voice command to identify objects in the scene"""
        if not self.current_detections:
            self.announce("I don't see any objects clearly right now.", priority=True)
            return "No objects detected"
            
        # Group objects by class
        objects_by_class = {}
        for obj in self.current_detections:
            class_name = obj['class']
            if class_name in objects_by_class:
                objects_by_class[class_name].append(obj)
            else:
                objects_by_class[class_name] = [obj]
                
        # Create announcement
        announcement = "I can see: "
        object_descriptions = []
        
        for class_name, objects in objects_by_class.items():
            count = len(objects)
            desc = f"{count} {class_name}{'s' if count > 1 else ''}"
            object_descriptions.append(desc)
            
        announcement += ", ".join(object_descriptions)
        self.announce(announcement, priority=True)
        return announcement
    
    def cmd_read_text(self):
        """Voice command to read text in the scene"""
        self.announce("Looking for text to read...", priority=True)
        # Set OCR to activate on next frame
        self.ocr_active = True
        self.last_ocr_time = 0
        return "OCR activated"
    
    def cmd_navigate(self):
        """Voice command for navigation assistance"""
        if self.safe_path:
            self.announce(f"Navigation guidance: {self.safe_path}", priority=True)
            return self.safe_path
        else:
            self.announce("I don't have enough information to provide navigation guidance right now.", priority=True)
            return "No navigation guidance available"
    
    def cmd_remember_location(self):
        """Voice command to remember the current location"""
        # For a real system, this would use GPS or other positioning
        # Here we're just using a placeholder
        timestamp = time.strftime("%H:%M:%S")
        landmark_name = f"Landmark {len(self.landmarks) + 1}"
        
        # Ask for a name
        self.announce("What would you like to call this location?", priority=True)
        # In a real system, we'd use speech recognition here
        # For now, just use the auto-generated name
        
        self.landmarks[landmark_name] = {
            'time': timestamp,
            'description': self.generate_scene_description(self.current_detections, self.depth_map)
        }
        
        self.announce(f"I've remembered this location as {landmark_name}", priority=True)
        return f"Location saved as {landmark_name}"
    
    def cmd_locate_landmark(self):
        """Voice command to find previously remembered landmarks"""
        if not self.landmarks:
            self.announce("You haven't saved any landmarks yet.", priority=True)
            return "No landmarks saved"
            
        # List available landmarks
        landmark_list = ", ".join(self.landmarks.keys())
        self.announce(f"You have saved these landmarks: {landmark_list}", priority=True)
        return "Landmarks listed"

    def cmd_help(self):
        """Voice command to list available commands"""
        help_text = ("Available commands: "
                    "describe - get a scene description, "
                    "identify - list objects in view, "
                    "read - detect and read text, "
                    "navigate - get navigation guidance, "
                    "remember - save current location, "
                    "locate - find saved landmarks, "
                    "help - list these commands")
        self.announce(help_text, priority=True)
        return help_text
    
    def enhance_image_for_ocr(self, image):
        """Apply preprocessing to enhance image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply dilation to make text more visible
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        return dilated
    
    def detect_crosswalk(self, frame):
        """Detect crosswalk patterns in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for patterns of parallel lines
        large_rect_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size and aspect ratio
            if w > 20 and h > 20 and 0.1 < w/h < 10:
                large_rect_contours.append(contour)
        
        # Count potential crosswalk lines
        if len(large_rect_contours) >= 3:
            # Analyze the pattern
            bounding_rects = [cv2.boundingRect(c) for c in large_rect_contours]
            
            # Sort by y-coordinate
            bounding_rects.sort(key=lambda r: r[1])
            
            # Check for roughly parallel lines
            parallel_count = 0
            for i in range(len(bounding_rects) - 1):
                r1 = bounding_rects[i]
                r2 = bounding_rects[i+1]
                
                # Check if heights are similar and they're horizontally aligned
                if abs(r1[3] - r2[3]) < 15 and abs(r1[1] - r2[1]) < 50:
                    parallel_count += 1
            
            if parallel_count >= 2:
                return True, bounding_rects
        
        return False, []
    
    def detect_traffic_signals(self, frame):
        """Detect traffic signal colors"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for traffic light colors
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([90, 255, 255])
        
        yellow_lower = np.array([15, 150, 150])
        yellow_upper = np.array([35, 255, 255])
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Count non-zero pixels in each mask
        red_count = cv2.countNonZero(red_mask)
        green_count = cv2.countNonZero(green_mask)
        yellow_count = cv2.countNonZero(yellow_mask)
        
        # Determine dominant color
        color_counts = {'red': red_count, 'green': green_count, 'yellow': yellow_count}
        dominant_color = max(color_counts, key=color_counts.get)
        
        # Only consider it a traffic signal if there's a significant amount of color
        threshold = 500  # Adjust based on testing
        if color_counts[dominant_color] > threshold:
            return True, dominant_color
        
        return False, None
    
    def detect_stairs(self, frame, depth_map):
        """Detect stairs using depth map patterns"""
        if depth_map is None:
            return False
            
        # Focus on the lower half of the image where stairs would typically be
        lower_half = depth_map[self.frame_height//2:, :]
        
        # Look for horizontal bands of similar depth
        # First, apply a median filter to smooth the depth map
        smoothed = cv2.medianBlur(np.float32(lower_half), 5)
        
        # Calculate vertical gradient
        gradient_y = np.diff(smoothed, axis=0)
        
        # Look for significant vertical changes
        threshold = 0.05  # Threshold for significant depth change
        significant_changes = np.abs(gradient_y) > threshold
        
        # Count rows with significant changes
        change_counts = np.sum(significant_changes, axis=1)
        
        # Look for rows with many changes (potential stair edges)
        potential_edges = change_counts > (self.frame_width // 4)
        
        # Check if we have multiple potential edges spaced appropriately
        edge_indices = np.where(potential_edges)[0]
        
        if len(edge_indices) >= 3:
            # Check if edges are regularly spaced (like stairs would be)
            spacing = np.diff(edge_indices)
            avg_spacing = np.mean(spacing)
            std_spacing = np.std(spacing)
            
            # Regular spacing would have low standard deviation
            if std_spacing < (avg_spacing * 0.3) and 5 < avg_spacing < 30:
                return True
        
        return False
    
    def detect_doorways(self, frame, depth_map):
        """Detect doorways using depth discontinuities"""
        if depth_map is None:
            return False, None
        
        # Use the depth map to find vertical structures
        # Apply Sobel filter to detect vertical edges
        sobelx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # Threshold the edge image
        _, binary = cv2.threshold(scaled_sobel, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours of vertical structures
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for tall rectangular structures
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's tall enough to be a door
            if h > self.frame_height * 0.5 and 0.15 < w / self.frame_width < 0.7:
                # Check depth context - doors should have a depth discontinuity
                if x > 0 and x + w < self.frame_width:
                    left_depth = np.median(depth_map[:, max(0, x-10):x])
                    right_depth = np.median(depth_map[:, x+w:min(self.frame_width, x+w+10)])
                    door_depth = np.median(depth_map[:, x:x+w])
                    
                    # Door should be at a different depth than surroundings
                    if abs(door_depth - left_depth) > 0.1 or abs(door_depth - right_depth) > 0.1:
                        door_center_x = x + w//2
                        direction = self.get_direction((x, 0, x+w, self.frame_height))
                        return True, (direction, door_depth * 10.0)  # Convert to meters
        
        return False, None
    
    def detect_sidewalk_edge(self, frame, depth_map):
        """Detect sidewalk edges using depth and color cues"""
        if depth_map is None:
            return False, None
            
        # Focus on the lower third of the image where sidewalk edges would be
        lower_third = depth_map[2*self.frame_height//3:, :]
        
        # Look for horizontal discontinuities in depth
        gradient_x = np.diff(lower_third, axis=1)
        
        # Find significant horizontal transitions
        threshold = 0.1
        significant_edges = np.abs(gradient_x) > threshold
        
        # Count the number of significant edges per column
        edge_counts = np.sum(significant_edges, axis=0)
        
        # Find columns with many edges
        edge_columns = np.where(edge_counts > (self.frame_height // 6))[0]
        
        if len(edge_columns) > 0:
            # Group adjacent columns
            edges = []
            current_edge = [edge_columns[0]]
            
            for i in range(1, len(edge_columns)):
                if edge_columns[i] - edge_columns[i-1] <= 5:  # Adjacent columns
                    current_edge.append(edge_columns[i])
                else:
                    edges.append(current_edge)
                    current_edge = [edge_columns[i]]
            
            if current_edge:
                edges.append(current_edge)
            
            # Only consider long enough edges
            significant_edges = [e for e in edges if len(e) > (self.frame_width // 10)]
            
            if significant_edges:
                # Find the most prominent edge
                longest_edge = max(significant_edges, key=len)
                edge_center = longest_edge[len(longest_edge) // 2]
                
                # Determine direction
                if edge_center < self.frame_width * 0.45:
                    direction = "to your left"
                elif edge_center > self.frame_width * 0.55:
                    direction = "to your right"
                else:
                    direction = "directly ahead"
                
                return True, direction
        
        return False, None
    
    def detect_human_pose(self, frame):
        """Detect human poses to recognize gestures or interactions"""
        try:
            # This requires additional model - placeholder for now
            # In a real implementation, we would use a pose detection model
            # For now, check if there are people detected by YOLO
            people = [obj for obj in self.current_detections if obj['class'] == 'person']
            
            if people:
                closest_person = min(people, key=lambda p: p['distance'] if p['distance'] else float('inf'))
                
                # Calculate if they might be facing the user
                box = closest_person['box']
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Use aspect ratio as a crude approximation of orientation
                aspect_ratio = width / height if height > 0 else 0
                
                if aspect_ratio > 0.5:  # Wider than tall - might be facing sideways
                    pose = "sideways"
                else:
                    pose = "facing you"
                
                return True, closest_person['distance'], pose
            
            return False, None, None
            
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return False, None, None
    
    def detect_currency(self, frame):
        """Identify currency notes - simplified placeholder"""
        # This is a placeholder for currency recognition
        # In a real implementation, we would use a specialized model
        
        # For demo purposes, let's use color/texture analysis
        try:
            # Convert to grayscale and examine texture patterns
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gabor filter to detect texture patterns common in currency
            ksize = 31
            sigma = 5
            theta = np.pi/4
            lambd = 10.0
            gamma = 0.5
            
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            
            # Check texture response
            mean_response = np.mean(filtered)
            
            # Simple thresholding for demonstration
            if mean_response > 50:
                # In a real system, we would identify the denomination
                return True, "currency note detected"
            
            return False, None
            
        except Exception as e:
            print(f"Error in currency detection: {e}")
            return False, None
    
    def analyze_crowd_density(self, frame):
        """Estimate the density of people in a scene"""
        people = [obj for obj in self.current_detections if obj['class'] == 'person']
        
        if not people:
            return "No people detected"
            
        # Count the number of people
        count = len(people)
        
        # Calculate average distance
        distances = [p['distance'] for p in people if p['distance']]
        if distances:
            avg_distance = sum(distances) / len(distances)
        else:
            avg_distance = None
            
        # Calculate density based on count and frame coverage
        total_area = self.frame_width * self.frame_height
        people_area = sum((obj['box'][2] - obj['box'][0]) * (obj['box'][3] - obj['box'][1]) for obj in people)
        
        density_ratio = people_area / total_area if total_area > 0 else 0
        
        # Determine crowdedness
        if count <= 2:
            density = "few people"
        elif count <= 5:
            density = "several people"
        elif count <= 10:
            density = "moderately crowded"
        else:
            density = "very crowded"
            
        if avg_distance is not None:
            proximity = f", nearest about {min(distances):.1f} meters away" if distances else ""
        else:
            proximity = ""
            
        return f"{count} people detected, {density}{proximity}"
    
    def detect_movement(self, prev_frame, curr_frame):
        """Detect movement in the scene between frames"""
        if prev_frame is None or curr_frame is None:
            return False, None
            
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Apply thresholding
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
        
        if significant_contours:
            # Determine where the most movement is occurring
            max_area_contour = max(significant_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_area_contour)
            
            # Determine direction of movement
            center_x = x + w//2
            center_y = y + h//2
            
            # Determine position relative to frame center
            if center_x < self.frame_width * 0.33:
                horizontal = "left"
            elif center_x > self.frame_width * 0.66:
                horizontal = "right"
            else:
                horizontal = "center"
                
            if center_y < self.frame_height * 0.33:
                vertical = "top"
            elif center_y > self.frame_height * 0.66:
                vertical = "bottom"
            else:
                vertical = "middle"
                
            return True, f"{vertical} {horizontal}"
        
        return False, None
    
    def detect_obstacle_type(self, obj_data, depth_map):
        """Classify obstacles by type and severity"""
        class_name = obj_data['class']
        distance = obj_data['distance']
        
        if not distance:
            return "unknown", 0
            
        # Default severity based on distance
        if distance < 1.0:
            base_severity = 3  # High
        elif distance < 2.5:
            base_severity = 2  # Medium
        else:
            base_severity = 1  # Low
            
        # Adjust severity based on object type
        high_risk_objects = ["car", "bicycle", "motorcycle", "truck", "bus", "fire hydrant", "bench"]
        medium_risk_objects = ["chair", "table", "pole", "traffic light", "potted plant"]
        low_risk_objects = ["backpack", "umbrella", "handbag"]
        
        if class_name in high_risk_objects:
            obstacle_type = "dangerous obstacle"
            severity = base_severity + 1
        elif class_name in medium_risk_objects:
            obstacle_type = "fixed obstacle"
            severity = base_severity
        elif class_name in low_risk_objects:
            obstacle_type = "small obstacle"
            severity = base_severity - 1
        else:
            obstacle_type = "obstacle"
            severity = base_severity
            
        # Cap severity between 1-3
        severity = max(1, min(3, severity))
        
        return obstacle_type, severity
    
    def map_surroundings(self):
        """Create a simplified spatial map of surroundings"""
        if not self.current_detections or self.depth_map is None:
            return "Insufficient data to map surroundings"
            
        # Divide the space into 9 sectors (3x3 grid)
        sectors = [[[] for _ in range(3)] for _ in range(3)]
        
        # Assign objects to sectors
        for obj in self.current_detections:
            box = obj['box']
            x1, y1, x2, y2 = box
            
            # Find center of the object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Determine sector
            sector_x = min(2, center_x * 3 // self.frame_width)
            sector_y = min(2, center_y * 3 // self.frame_height)
            
            # Add to appropriate sector
            sectors[sector_y][sector_x].append(obj)
        
        # Create a textual map
        sector_names = [
            ["top left", "top center", "top right"],
            ["left", "center", "right"],
            ["bottom left", "bottom center", "bottom right"]
        ]
        
        # Generate map description
        description = "Spatial map: "
        for y in range(3):
            for x in range(3):
                if sectors[y][x]:
                    # Count objects by type
                    obj_counts = {}
                    for obj in sectors[y][x]:
                        class_name = obj['class']
                        if class_name in obj_counts:
                            obj_counts[class_name] += 1
                        else:
                            obj_counts[class_name] = 1
                    
                    # Create description for this sector
                    items = []
                    for class_name, count in obj_counts.items():
                        items.append(f"{count} {class_name}{'s' if count > 1 else ''}")
                    
                    sector_desc = f"In {sector_names[y][x]}: {', '.join(items)}. "
                    description += sector_desc
        
        return description
    
    def recognize_environment(self):
        """Identify the type of environment based on detected objects"""
        if not self.current_detections:
            return "Unknown environment"
            
        # Define environment indicators
        environments = {
            "kitchen": ["refrigerator", "microwave", "oven", "sink", "bowl", "cup", "bottle", "knife", "spoon", "fork"],
            "living room": ["sofa", "chair", "tv", "remote", "book", "vase", "potted plant"],
            "bathroom": ["toilet", "sink", "toothbrush", "hair drier"],
            "bedroom": ["bed", "chair", "clock"],
            "office": ["chair", "desk", "laptop", "keyboard", "mouse", "monitor"],
            "street": ["car", "truck", "bus", "traffic light", "stop sign", "bicycle", "motorcycle", "person"],
            "store": ["bottle", "person", "chair", "refrigerator", "vase"],
            "park": ["bench", "bicycle", "dog", "person", "potted plant", "tree"]
        }
        
        # Count occurrences of each environment's indicators
        env_scores = {env: 0 for env in environments}
        
        # Get all detected classes
        detected_classes = [obj['class'] for obj in self.current_detections]
        
        # Score each environment
        for env, indicators in environments.items():
            for indicator in indicators:
                env_scores[env] += detected_classes.count(indicator)
        
        # Find environment with highest score
        if max(env_scores.values()) > 0:
            likely_env = max(env_scores, key=env_scores.get)
            confidence = min(100, env_scores[likely_env] * 20)  # Simple confidence score
            return f"{likely_env} ({confidence}% confidence)"
        else:
            return "Unknown environment"
    
    def run(self):
        """Run the system's main loop"""
        prev_frame = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            try:
                # Process the frame with all our detection methods
                processed_frame = self.process_frame(frame)
                
                # Perform additional detections
                
                # Check for crosswalks
                crosswalk_detected, _ = self.detect_crosswalk(frame)
                if crosswalk_detected:
                    cv2.putText(processed_frame, "Crosswalk detected", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if time.time() - self.last_speech_time > self.speech_cooldown:
                        self.announce("Crosswalk detected ahead")
                        self.last_speech_time = time.time()
                
                # Check for traffic signals
                signal_detected, color = self.detect_traffic_signals(frame)
                if signal_detected:
                    cv2.putText(processed_frame, f"Traffic signal: {color}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if time.time() - self.last_speech_time > self.speech_cooldown and color in ["red", "green"]:
                        self.announce(f"Traffic light is {color}")
                        self.last_speech_time = time.time()
                
                # Check for stairs
                if self.depth_map is not None:
                    stairs_detected = self.detect_stairs(frame, self.depth_map)
                    if stairs_detected:
                        cv2.putText(processed_frame, "Stairs detected", (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if time.time() - self.last_speech_time > self.speech_cooldown:
                            self.announce("Warning: Stairs detected ahead")
                            self.play_sound("proximity_alert")
                            self.last_speech_time = time.time()
                
                # Check for doorways
                doorway_detected, door_info = self.detect_doorways(frame, self.depth_map)
                if doorway_detected:
                    direction, distance = door_info
                    cv2.putText(processed_frame, f"Doorway {direction}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if time.time() - self.last_speech_time > self.speech_cooldown:
                        self.announce(f"Doorway detected {direction}, {distance:.1f} meters away")
                        self.last_speech_time = time.time()
                
                # Detect sidewalk edges
                sidewalk_detected, sidewalk_direction = self.detect_sidewalk_edge(frame, self.depth_map)
                if sidewalk_detected:
                    cv2.putText(processed_frame, f"Sidewalk edge {sidewalk_direction}", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    if time.time() - self.last_speech_time > self.speech_cooldown:
                        self.announce(f"Sidewalk edge {sidewalk_direction}")
                        self.last_speech_time = time.time()
                
                # Detect human interactions
                human_detected, human_distance, human_pose = self.detect_human_pose(frame)
                if human_detected and human_distance and human_distance < 3.0:
                    cv2.putText(processed_frame, f"Person {human_pose}, {human_distance:.1f}m", (10, 210),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)
                    
                # Detect movement between frames
                if prev_frame is not None:
                    movement_detected, movement_location = self.detect_movement(prev_frame, frame)
                    if movement_detected:
                        cv2.putText(processed_frame, f"Movement: {movement_location}", (10, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Detect emotions in faces
                emotions = self.detect_emotions(frame)
                for emotion_data in emotions:
                    x, y, w, h = emotion_data['face_box']
                    emotion = emotion_data['emotion']
                    confidence = emotion_data['confidence']
                    
                    # Draw emotion information
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.putText(processed_frame, f"{emotion} ({confidence:.0%})", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    # Announce if confidence is high
                    if confidence > 0.7 and time.time() - self.last_speech_time > self.speech_cooldown:
                        self.announce(f"Person {emotion.lower()}")
                        self.last_speech_time = time.time()
                
                # Detect INR currency notes
                current_time = time.time()
                if current_time - self.last_currency_detection_time > self.currency_cooldown:
                    detected_notes, currency_count, total_value = self.detect_inr_currency(frame)
                    
                    if detected_notes:
                        # Draw currency information
                        for note in detected_notes:
                            x1, y1, x2, y2 = note['box']
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(processed_frame, f"₹{note['denomination']}", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Announce currency count
                        if currency_count and time.time() - self.last_speech_time > self.speech_cooldown:
                            count_text = ", ".join([f"{count} ₹{denom} notes" for denom, count in currency_count.items()])
                            total_text = f"Total: ₹{total_value}"
                            self.announce(f"Currency detected: {count_text}. {total_text}")
                            self.last_speech_time = time.time()
                            self.last_currency_detection_time = current_time
                
                # Store current frame for next iteration
                prev_frame = frame.copy()
                
                # Environment recognition (less frequent)
                if time.time() - self.last_scene_time > self.scene_description_cooldown:
                    env_type = self.recognize_environment()
                    cv2.putText(processed_frame, f"Environment: {env_type}", (10, self.frame_height - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
                
                # Display the processed frame
                cv2.imshow('Enhanced Nayan', processed_frame)
                
                # Exit on ESC key
                if cv2.waitKey(1) == 27:
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        nayan = EnhancedNayan()
        nayan.announce("Enhanced Nayan system initialized")
        nayan.run()
    except Exception as e:
        print(f"Error initializing Nayan: {e}")