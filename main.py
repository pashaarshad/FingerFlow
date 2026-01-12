"""
FingerFlow - Hand Gesture Cursor Control
Control your mouse cursor using hand gestures via webcam

Version 2.0: Added Double Tap, Drag & Drop, Scroll
Uses MediaPipe Tasks API (0.10.x+)
"""

import cv2
import pyautogui
import numpy as np
import time
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math


class GestureDetector:
    """Detects specific hand gestures for actions"""
    
    def __init__(self):
        # Tap detection (single and double click)
        self.last_pinch_time = 0
        self.pinch_count = 0
        self.double_tap_threshold = 0.35  # seconds between taps for double-click
        self.click_delay = 0.25  # wait time before registering single click
        self.pinch_threshold = 40  # pixels distance for pinch
        self.was_pinched = False
        self.pending_single_click = False
        self.pending_click_time = 0
        
        # Drag detection
        self.is_dragging = False
        self.drag_start_pos = None
        
        # Scroll detection
        self.scroll_mode = False
        self.last_scroll_y = None
        self.scroll_sensitivity = 3  # pixels to trigger scroll
        
    def calculate_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_pinch(self, landmarks):
        """Detect if thumb and index finger are pinched together"""
        if len(landmarks) < 21:
            return False
        
        # Get thumb tip (4) and index tip (8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = self.calculate_distance(
            (thumb_tip[1], thumb_tip[2]), 
            (index_tip[1], index_tip[2])
        )
        
        return distance < self.pinch_threshold
    
    def detect_click(self, landmarks):
        """
        Detect single or double click based on pinch gesture.
        
        Returns:
            - "single_click" if single pinch detected
            - "double_click" if two quick pinches detected
            - None if no click action
        """
        is_pinched = self.detect_pinch(landmarks)
        current_time = time.time()
        
        # Detect pinch START (transition from not pinched to pinched)
        if is_pinched and not self.was_pinched:
            time_since_last_pinch = current_time - self.last_pinch_time
            
            if time_since_last_pinch < self.double_tap_threshold:
                # Second pinch within threshold = double click!
                self.pinch_count = 0
                self.pending_single_click = False
                self.was_pinched = is_pinched
                self.last_pinch_time = current_time
                return "double_click"
            else:
                # First pinch - mark as pending single click
                self.pinch_count = 1
                self.pending_single_click = True
                self.pending_click_time = current_time
            
            self.last_pinch_time = current_time
        
        # Detect pinch RELEASE (transition from pinched to not pinched)
        elif not is_pinched and self.was_pinched:
            # Pinch released, check if we should trigger single click
            pass
        
        self.was_pinched = is_pinched
        
        # Check if pending single click should be triggered
        # (waited long enough without second pinch)
        if self.pending_single_click and not is_pinched:
            if current_time - self.pending_click_time > self.click_delay:
                self.pending_single_click = False
                self.pinch_count = 0
                return "single_click"
        
        return None

    
    def detect_grab(self, fingers_status):
        """Detect grab gesture (closed fist - all fingers down)"""
        if len(fingers_status) != 5:
            return False
        
        # All fingers should be down (closed fist)
        # Thumb can be flexible, check other 4 fingers
        return sum(fingers_status[1:]) <= 1  # At most 1 finger up
    
    def detect_open_hand(self, fingers_status):
        """Detect open hand gesture (all fingers up)"""
        if len(fingers_status) != 5:
            return False
        
        # At least 4 fingers should be up
        return sum(fingers_status) >= 4
    
    def detect_two_fingers_scroll(self, landmarks, fingers_status):
        """Detect two-finger scroll gesture (index + middle extended)
        
        Scroll is triggered by the MOVEMENT of fingers (like a swipe), 
        not by moving the whole hand. Even if hand stays in same position,
        moving fingers up/down will scroll.
        """
        if len(fingers_status) != 5 or len(landmarks) < 21:
            self.last_scroll_y = None
            self.scroll_history = []
            return None, 0
        
        # Check if only index (1) and middle (2) fingers are up
        # fingers_status: [thumb, index, middle, ring, pinky]
        index_up = fingers_status[1] == 1
        middle_up = fingers_status[2] == 1
        ring_down = fingers_status[3] == 0
        pinky_down = fingers_status[4] == 0
        
        is_scroll_gesture = index_up and middle_up and ring_down and pinky_down
        
        if is_scroll_gesture:
            # Get average Y position of index and middle finger tips
            index_tip_y = landmarks[8][2]  # (idx, x, y) -> y
            middle_tip_y = landmarks[12][2]
            current_y = (index_tip_y + middle_tip_y) / 2
            
            # Initialize scroll history if needed
            if not hasattr(self, 'scroll_history'):
                self.scroll_history = []
            
            if self.last_scroll_y is not None:
                # Calculate movement delta
                delta_y = current_y - self.last_scroll_y
                
                # Add to history for smoothing (keep last 3 frames)
                self.scroll_history.append(delta_y)
                if len(self.scroll_history) > 3:
                    self.scroll_history.pop(0)
                
                # Calculate average movement direction
                avg_delta = sum(self.scroll_history) / len(self.scroll_history)
                
                # Trigger scroll if movement is significant (lowered threshold)
                if abs(avg_delta) > 2:  # Reduced sensitivity threshold
                    # Scale the scroll amount based on movement speed
                    scroll_amount = int(avg_delta / 2)
                    if scroll_amount != 0:
                        self.last_scroll_y = current_y
                        return "scroll", scroll_amount
            
            self.last_scroll_y = current_y
            return "scroll_mode", 0
        else:
            self.last_scroll_y = None
            return None, 0
    
    def update_drag_state(self, fingers_status, cursor_pos):
        """Update drag state based on hand gesture"""
        is_grabbing = self.detect_grab(fingers_status)
        is_open = self.detect_open_hand(fingers_status)
        
        if is_grabbing and not self.is_dragging:
            # Start dragging
            self.is_dragging = True
            self.drag_start_pos = cursor_pos
            return "drag_start"
        elif is_open and self.is_dragging:
            # Stop dragging (drop)
            self.is_dragging = False
            self.drag_start_pos = None
            return "drag_end"
        elif self.is_dragging:
            return "dragging"
        
        return None


class HandTracker:
    """Handles hand detection and landmark tracking using MediaPipe Tasks API"""
    
    # Hand connection pairs for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17), (0, 17)  # Palm
    ]
    
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        # Model path - will download if needed
        self.model_path = self._get_model_path()
        
        # Create hand landmarker options
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.results = None
        
        # Landmark indices
        self.INDEX_TIP = 8
        self.THUMB_TIP = 4
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
    def _get_model_path(self):
        """Download or get the hand landmarker model"""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")
        
        return model_path
    
    def _draw_landmarks(self, frame, hand_landmarks):
        """Custom drawing function for hand landmarks"""
        h, w, _ = frame.shape
        
        # Get pixel coordinates for all landmarks
        points = []
        for landmark in hand_landmarks:
            px, py = int(landmark.x * w), int(landmark.y * h)
            points.append((px, py))
        
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        
        # Draw landmarks
        for i, (px, py) in enumerate(points):
            # Different colors for different parts
            if i == 0:  # Wrist
                color = (255, 0, 255)  # Magenta
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue
            
            cv2.circle(frame, (px, py), 5, color, cv2.FILLED)
            cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)
        
    def find_hands(self, frame, draw=True):
        """Detect hands in frame and optionally draw landmarks"""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        self.results = self.detector.detect(mp_image)
        
        # Draw landmarks if requested
        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks)
        
        return frame
    
    def get_landmark_position(self, frame, hand_index=0, landmark_id=8):
        """Get the position of a specific landmark"""
        h, w, _ = frame.shape
        
        if self.results and self.results.hand_landmarks:
            if hand_index < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_index]
                landmark = hand[landmark_id]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                return cx, cy, True
        
        return 0, 0, False
    
    def get_all_landmarks(self, frame, hand_index=0):
        """Get all 21 landmark positions"""
        landmarks = []
        h, w, _ = frame.shape
        
        if self.results and self.results.hand_landmarks:
            if hand_index < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_index]
                for idx, landmark in enumerate(hand):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((idx, cx, cy))
        
        return landmarks
    
    def fingers_up(self, frame, hand_index=0):
        """Check which fingers are up (extended)"""
        fingers = []
        landmarks = self.get_all_landmarks(frame, hand_index)
        
        if len(landmarks) == 0:
            return []
        
        # Convert to dict for easy access
        lm_dict = {idx: (x, y) for idx, x, y in landmarks}
        
        # Thumb (compare x positions for left/right hand detection)
        if lm_dict[4][0] > lm_dict[3][0]:  # Simplified - assumes right hand
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other 4 fingers - tip should be above pip (lower y = higher on screen)
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        
        for tip, pip in zip(tip_ids, pip_ids):
            if lm_dict[tip][1] < lm_dict[pip][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def release(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'detector'):
            self.detector.close()


class CursorController:
    """Handles smooth cursor movement and screen coordinate mapping
    
    Supports two modes:
    - DIRECT: Finger position maps directly to screen position
    - JOYSTICK: Finger direction controls cursor velocity (like a joystick)
    """
    
    def __init__(self, smoothing=7, mode="joystick"):
        # Get screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Disable PyAutoGUI fail-safe for smoother operation
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        # Control mode: "direct" or "joystick"
        self.mode = mode
        
        # Smoothing buffer (increased for smoother movement)
        self.smoothing = smoothing
        self.position_buffer = []
        
        # Current cursor position (for joystick mode)
        self.cursor_x = self.screen_w // 2
        self.cursor_y = self.screen_h // 2
        
        # Joystick mode settings
        self.dead_zone = 0.15  # Center zone where cursor doesn't move
        self.max_speed = 25  # Maximum cursor movement speed (pixels per frame)
        self.acceleration = 1.5  # How quickly cursor accelerates
        
        # Frame reduction zone (percentage from edges to ignore)
        self.frame_reduction = 0.1
        
        # Reference center point (will be set when hand is first detected)
        self.center_x = None
        self.center_y = None
        self.calibrated = False
        
    def calibrate_center(self, x, y, frame_w, frame_h):
        """Set the center reference point for joystick mode"""
        self.center_x = frame_w // 2
        self.center_y = frame_h // 2
        self.calibrated = True
        
    def map_to_screen_direct(self, x, y, frame_w, frame_h):
        """Direct mapping: finger position = screen position"""
        # Calculate the active zone
        x_min = int(frame_w * self.frame_reduction)
        x_max = int(frame_w * (1 - self.frame_reduction))
        y_min = int(frame_h * self.frame_reduction)
        y_max = int(frame_h * (1 - self.frame_reduction))
        
        # Clamp values to active zone
        x = max(x_min, min(x, x_max))
        y = max(y_min, min(y, y_max))
        
        # Map to screen coordinates
        screen_x = np.interp(x, [x_min, x_max], [0, self.screen_w])
        screen_y = np.interp(y, [y_min, y_max], [0, self.screen_h])
        
        return int(screen_x), int(screen_y)
    
    def map_to_screen_joystick(self, x, y, frame_w, frame_h):
        """Joystick mapping: finger direction controls cursor velocity"""
        if not self.calibrated:
            self.calibrate_center(x, y, frame_w, frame_h)
        
        # Calculate offset from center (normalized -1 to 1)
        offset_x = (x - self.center_x) / (frame_w * 0.4)  # 40% of frame width = full speed
        offset_y = (y - self.center_y) / (frame_h * 0.4)
        
        # Clamp to -1 to 1
        offset_x = max(-1, min(1, offset_x))
        offset_y = max(-1, min(1, offset_y))
        
        # Apply dead zone (center area where cursor doesn't move)
        if abs(offset_x) < self.dead_zone:
            offset_x = 0
        else:
            # Rescale to remove dead zone
            offset_x = (offset_x - np.sign(offset_x) * self.dead_zone) / (1 - self.dead_zone)
        
        if abs(offset_y) < self.dead_zone:
            offset_y = 0
        else:
            offset_y = (offset_y - np.sign(offset_y) * self.dead_zone) / (1 - self.dead_zone)
        
        # Apply acceleration curve (exponential for finer control)
        velocity_x = np.sign(offset_x) * (abs(offset_x) ** self.acceleration) * self.max_speed
        velocity_y = np.sign(offset_y) * (abs(offset_y) ** self.acceleration) * self.max_speed
        
        # Update cursor position
        self.cursor_x += velocity_x
        self.cursor_y += velocity_y
        
        # Clamp to screen bounds
        self.cursor_x = max(0, min(self.screen_w - 1, self.cursor_x))
        self.cursor_y = max(0, min(self.screen_h - 1, self.cursor_y))
        
        return int(self.cursor_x), int(self.cursor_y)
    
    def map_to_screen(self, x, y, frame_w, frame_h):
        """Map camera coordinates to screen coordinates based on current mode"""
        if self.mode == "joystick":
            return self.map_to_screen_joystick(x, y, frame_w, frame_h)
        else:
            return self.map_to_screen_direct(x, y, frame_w, frame_h)
    
    def smooth_position(self, x, y):
        """Apply smoothing to reduce jitter"""
        self.position_buffer.append((x, y))
        
        if len(self.position_buffer) > self.smoothing:
            self.position_buffer.pop(0)
        
        # Calculate weighted average (recent positions weighted more)
        total_weight = 0
        avg_x = 0
        avg_y = 0
        for i, (px, py) in enumerate(self.position_buffer):
            weight = i + 1  # Later positions get more weight
            avg_x += px * weight
            avg_y += py * weight
            total_weight += weight
        
        avg_x = int(avg_x / total_weight)
        avg_y = int(avg_y / total_weight)
        
        return avg_x, avg_y
    
    def move_cursor(self, x, y):
        """Move cursor to position with smoothing"""
        smooth_x, smooth_y = self.smooth_position(x, y)
        pyautogui.moveTo(smooth_x, smooth_y)
        return smooth_x, smooth_y
    
    def click(self):
        """Perform a single click"""
        pyautogui.click()
    
    def double_click(self):
        """Perform a double click"""
        pyautogui.doubleClick()
    
    def start_drag(self):
        """Start dragging (mouse down)"""
        pyautogui.mouseDown()
    
    def end_drag(self):
        """End dragging (mouse up)"""
        pyautogui.mouseUp()
    
    def scroll(self, amount):
        """Scroll by amount (negative = up, positive = down)"""
        pyautogui.scroll(-amount)  # Invert for natural scroll direction


def draw_status_panel(frame, gesture_status, is_dragging, scroll_mode):
    """Draw a status panel showing current gesture state"""
    h, w, _ = frame.shape
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 200, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw status text
    cv2.putText(frame, "GESTURES:", (w - 190, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Drag status
    drag_color = (0, 255, 0) if is_dragging else (128, 128, 128)
    drag_text = "DRAGGING" if is_dragging else "Drag: Ready"
    cv2.putText(frame, drag_text, (w - 190, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, drag_color, 1)
    
    # Scroll status
    scroll_color = (0, 255, 255) if scroll_mode else (128, 128, 128)
    scroll_text = "SCROLL MODE" if scroll_mode else "Scroll: Ready"
    cv2.putText(frame, scroll_text, (w - 190, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, scroll_color, 1)
    
    # Last gesture
    if gesture_status:
        cv2.putText(frame, f"Last: {gesture_status}", (w - 190, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_joystick_guide(frame, finger_x, finger_y, controller):
    """Draw a visual joystick guide showing center and finger direction"""
    h, w, _ = frame.shape
    center_x = w // 2
    center_y = h // 2
    
    # Draw center crosshair (dead zone indicator)
    dead_zone_radius = int(w * controller.dead_zone * 0.4)
    
    # Draw dead zone circle (semi-transparent)
    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), dead_zone_radius, (100, 100, 100), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw dead zone outline
    cv2.circle(frame, (center_x, center_y), dead_zone_radius, (200, 200, 200), 2)
    
    # Draw center point
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
    
    # Draw crosshairs
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (150, 150, 150), 1)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (150, 150, 150), 1)
    
    # Draw direction line from center to finger
    if finger_x is not None and finger_y is not None:
        # Calculate distance from center
        dist = math.sqrt((finger_x - center_x)**2 + (finger_y - center_y)**2)
        
        if dist > dead_zone_radius:
            # Draw line showing direction
            cv2.line(frame, (center_x, center_y), (finger_x, finger_y), (0, 255, 0), 2)
            
            # Draw arrow head
            angle = math.atan2(finger_y - center_y, finger_x - center_x)
            arrow_len = 15
            cv2.arrowedLine(frame, 
                           (center_x, center_y), 
                           (finger_x, finger_y),
                           (0, 255, 0), 2, tipLength=0.15)
    
    # Draw mode label
    cv2.putText(frame, "JOYSTICK MODE", (center_x - 60, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    """Main application loop"""
    print("=" * 60)
    print("   FingerFlow - Hand Gesture Cursor Control v2.1")
    print("=" * 60)
    print("\n🕹️ JOYSTICK MODE: Point finger in direction to move cursor")
    print("   - Center = Stop | Tilt direction = Move that way")
    print("   - Small movements = Slow | Large movements = Fast")
    print("\nGestures:")
    print("  - INDEX FINGER: Control cursor direction")
    print("  - PINCH (thumb + index) x1: Single click")
    print("  - PINCH (thumb + index) x2: Double click")
    print("  - CLOSED FIST: Start drag")
    print("  - OPEN HAND: Drop (release drag)")
    print("  - TWO FINGERS (index + middle) UP/DOWN: Scroll")
    print("\n  - Press 'Q' to quit")
    print("\nStarting camera...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Initialize components
    tracker = HandTracker(max_hands=1, detection_confidence=0.7, tracking_confidence=0.7)
    controller = CursorController(smoothing=5)
    gesture_detector = GestureDetector()
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    
    # Status tracking
    last_gesture = ""
    scroll_active = False
    
    print("Camera started! Use hand gestures to control the cursor.\n")
    
    try:
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            
            # Detect hands
            frame = tracker.find_hands(frame, draw=True)
            
            # Get landmarks and finger status
            landmarks = tracker.get_all_landmarks(frame)
            fingers = tracker.fingers_up(frame)
            
            # Get index finger position for cursor
            x, y, found = tracker.get_landmark_position(frame, hand_index=0, landmark_id=tracker.INDEX_TIP)
            
            if found and len(landmarks) >= 21:
                # Draw a larger circle on index finger tip
                cv2.circle(frame, (x, y), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
                
                # Map to screen and move cursor
                screen_x, screen_y = controller.map_to_screen(x, y, frame_w, frame_h)
                
                # Check for scroll gesture first (two fingers)
                scroll_action, scroll_amount = gesture_detector.detect_two_fingers_scroll(landmarks, fingers)
                
                if scroll_action == "scroll" and scroll_amount != 0:
                    controller.scroll(scroll_amount)
                    last_gesture = f"Scroll {'Up' if scroll_amount < 0 else 'Down'}"
                    scroll_active = True
                elif scroll_action == "scroll_mode":
                    scroll_active = True
                else:
                    scroll_active = False
                    
                    # Check for drag gesture
                    drag_state = gesture_detector.update_drag_state(fingers, (screen_x, screen_y))
                    
                    if drag_state == "drag_start":
                        controller.start_drag()
                        last_gesture = "Drag Start"
                        print("🖐️ Drag started!")
                    elif drag_state == "drag_end":
                        controller.end_drag()
                        last_gesture = "Dropped!"
                        print("✋ Dropped!")
                    elif drag_state == "dragging":
                        # Continue moving while dragging
                        pass
                    else:
                        # Check for click gestures (only when not dragging/scrolling)
                        click_action = gesture_detector.detect_click(landmarks)
                        
                        if click_action == "single_click":
                            controller.click()
                            last_gesture = "Click!"
                            print("👆 Clicked!")
                        elif click_action == "double_click":
                            controller.double_click()
                            last_gesture = "Double Click!"
                            print("👆👆 Double clicked!")
                
                # Move cursor (always, for both normal and drag mode)
                final_x, final_y = controller.move_cursor(screen_x, screen_y)
                
                # Display cursor position
                cv2.putText(frame, f"Cursor: ({final_x}, {final_y})", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display finger status
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                finger_str = " ".join([f"{n[0]}:{f}" for n, f in zip(finger_names, fingers)])
                cv2.putText(frame, f"Fingers: {finger_str}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display status
            status = "Hand Detected" if found else "No Hand Detected"
            color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(frame, status, (10, frame_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw gesture status panel
            draw_status_panel(frame, last_gesture, gesture_detector.is_dragging, scroll_active)
            
            # Draw joystick guide if in joystick mode
            if controller.mode == "joystick":
                finger_x = x if found else None
                finger_y = y if found else None
                draw_joystick_guide(frame, finger_x, finger_y, controller)
            
            # Show frame
            cv2.imshow("FingerFlow v2.1 - Joystick Mode", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting FingerFlow...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user...")
    finally:
        # Ensure drag is released
        if gesture_detector.is_dragging:
            controller.end_drag()
        
        # Cleanup
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


if __name__ == "__main__":
    main()
