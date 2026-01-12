"""
FingerFlow - Hand Gesture Cursor Control
Control your mouse cursor using hand gestures via webcam

Version 1.0: Basic cursor movement with index finger
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
    """Handles smooth cursor movement and screen coordinate mapping"""
    
    def __init__(self, smoothing=5):
        # Get screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Disable PyAutoGUI fail-safe for smoother operation
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        # Smoothing buffer
        self.smoothing = smoothing
        self.prev_x, self.prev_y = 0, 0
        self.position_buffer = []
        
        # Frame reduction zone (percentage from edges to ignore)
        self.frame_reduction = 0.15
        
    def map_to_screen(self, x, y, frame_w, frame_h):
        """Map camera coordinates to screen coordinates with zone reduction"""
        # Calculate the active zone
        x_min = int(frame_w * self.frame_reduction)
        x_max = int(frame_w * (1 - self.frame_reduction))
        y_min = int(frame_h * self.frame_reduction)
        y_max = int(frame_h * (1 - self.frame_reduction))
        
        # Clamp values to active zone
        x = max(x_min, min(x, x_max))
        y = max(y_min, min(y, y_max))
        
        # Map to screen coordinates (mirror x-axis for natural movement)
        screen_x = np.interp(x, [x_min, x_max], [self.screen_w, 0])
        screen_y = np.interp(y, [y_min, y_max], [0, self.screen_h])
        
        return int(screen_x), int(screen_y)
    
    def smooth_position(self, x, y):
        """Apply smoothing to reduce jitter"""
        self.position_buffer.append((x, y))
        
        if len(self.position_buffer) > self.smoothing:
            self.position_buffer.pop(0)
        
        # Calculate average position
        avg_x = int(sum(p[0] for p in self.position_buffer) / len(self.position_buffer))
        avg_y = int(sum(p[1] for p in self.position_buffer) / len(self.position_buffer))
        
        return avg_x, avg_y
    
    def move_cursor(self, x, y):
        """Move cursor to position with smoothing"""
        smooth_x, smooth_y = self.smooth_position(x, y)
        pyautogui.moveTo(smooth_x, smooth_y)
        return smooth_x, smooth_y


def main():
    """Main application loop"""
    print("=" * 50)
    print("   FingerFlow - Hand Gesture Cursor Control")
    print("=" * 50)
    print("\nControls:")
    print("  - Move your INDEX FINGER to control the cursor")
    print("  - Press 'Q' to quit")
    print("\nStarting camera...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Initialize tracker and controller
    tracker = HandTracker(max_hands=1, detection_confidence=0.7, tracking_confidence=0.7)
    controller = CursorController(smoothing=5)
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    
    print("Camera started! Move your hand to control the cursor.\n")
    
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
            
            # Get index finger position
            x, y, found = tracker.get_landmark_position(frame, hand_index=0, landmark_id=tracker.INDEX_TIP)
            
            if found:
                # Draw a larger circle on index finger tip
                cv2.circle(frame, (x, y), 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
                
                # Map to screen and move cursor
                screen_x, screen_y = controller.map_to_screen(x, y, frame_w, frame_h)
                final_x, final_y = controller.move_cursor(screen_x, screen_y)
                
                # Display cursor position
                cv2.putText(frame, f"Cursor: ({final_x}, {final_y})", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
            
            # Show frame
            cv2.imshow("FingerFlow - Hand Gesture Control", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting FingerFlow...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user...")
    finally:
        # Cleanup
        tracker.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


if __name__ == "__main__":
    main()
