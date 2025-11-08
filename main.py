"""
Hue Hand Control - Control Philips Hue lights using hand gestures.

This application tracks thumb and index finger tips using MediaPipe,
calculates the distance between them, and uses that distance to control
the brightness of Philips Hue lights.
"""

import cv2
import mediapipe as mp
import numpy as np
from phue import Bridge
import math
import argparse
import sys
import time
from typing import Optional, Tuple


class HandTracker:
    """Hand tracking using MediaPipe for fingertip detection."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_landmarks(self, frame):
        """Detect hand landmarks in the frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
    
    def get_thumb_index_tips(self, landmarks, frame_shape) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Extract thumb and index finger tip coordinates.
        Returns ((thumb_x, thumb_y), (index_x, index_y)) or None if not detected.
        """
        if not landmarks:
            return None
        
        # MediaPipe hand landmark indices
        # Thumb tip: 4, Index finger tip: 8
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = frame_shape[:2]
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        
        return thumb_pos, index_pos
    
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on the frame."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS
            )


class HueController:
    """Controller for Philips Hue lights."""
    
    def __init__(self, bridge_ip: str, light_ids: list):
        """
        Initialize Hue Bridge connection.
        
        Args:
            bridge_ip: IP address of the Philips Hue Bridge
            light_ids: List of light IDs or names to control (e.g., [1, 2, 3] or ["Living Room", "Bedroom"])
        """
        self.bridge = Bridge(bridge_ip)
        self.light_ids = light_ids
        self.light_names = {}  # Store light names for display
        self.light_objects = []  # Store Light objects for direct control
        self.last_brightness = None  # Track last brightness to avoid redundant updates
        self.last_update_time = 0  # Track last update time for rate limiting
        self._connect()
        self._load_light_names()
        self._load_light_objects()
        
    def _connect(self):
        """Connect to the Hue Bridge (press button on first run)."""
        try:
            # This will work if already authenticated
            self.bridge.get_api()
        except Exception:
            print("Please press the button on your Hue Bridge now...")
            print("Waiting 10 seconds for you to press the button...")
            import time
            time.sleep(10)
            self.bridge.connect()
    
    def _load_light_names(self):
        """Load light names from the bridge for display purposes."""
        try:
            lights = self.bridge.get_light()
            for light_id, light_info in lights.items():
                self.light_names[str(light_id)] = light_info['name']
                # Also store by name for reverse lookup
                self.light_names[light_info['name']] = light_info['name']
        except Exception as e:
            print(f"Warning: Could not load light names: {e}")
    
    def _load_light_objects(self):
        """Load Light objects for direct control."""
        self.light_objects = []
        for light_id in self.light_ids:
            try:
                # Bridge can be accessed like a dict: bridge[light_id] or bridge['name']
                light_obj = self.bridge[light_id]
                self.light_objects.append(light_obj)
                print(f"Loaded light object: {light_obj.name} (ID: {light_id})")
            except Exception as e:
                print(f"Warning: Could not load light object for {light_id}: {e}")
    
    def _is_light_reachable(self, light_obj):
        """Check if a light is reachable by the bridge."""
        try:
            return light_obj.reachable
        except Exception:
            return False
    
    def get_light_display_name(self, light_id_or_name):
        """Get display name for a light (returns name if available, otherwise returns ID)."""
        return self.light_names.get(str(light_id_or_name), str(light_id_or_name))
            
    def set_brightness(self, brightness: int, min_update_interval: float = 0.1):
        """
        Set brightness for all configured lights.
        
        Args:
            brightness: Brightness value (1-254, where 1 is minimum, 254 is maximum)
            min_update_interval: Minimum time (seconds) between updates to avoid spamming the bridge
        """
        # Clamp brightness to valid range
        brightness = max(1, min(254, int(brightness)))
        
        # Rate limiting: only update if brightness changed or enough time has passed
        current_time = time.time()
        if (self.last_brightness == brightness and 
            current_time - self.last_update_time < min_update_interval):
            return brightness
        
        # Update only if brightness changed significantly (avoid jitter)
        if self.last_brightness is not None and abs(self.last_brightness - brightness) < 2:
            return brightness
        
        # Use Light objects for more reliable control
        for light_obj in self.light_objects:
            try:
                # Check if light is reachable
                if not self._is_light_reachable(light_obj):
                    print(f"Warning: Light {light_obj.name} is not reachable")
                    continue
                
                # Turn on the light first (if not already on)
                if not light_obj.on:
                    light_obj.on = True
                
                # Set brightness using the Light object property
                # This is more reliable than using set_light
                light_obj.brightness = brightness
                
            except Exception as e:
                print(f"Exception setting brightness for light {light_obj.name}: {e}")
                import traceback
                traceback.print_exc()
        
        self.last_brightness = brightness
        self.last_update_time = current_time
        return brightness


class GestureBrightnessMapper:
    """Maps finger distance to brightness values."""
    
    def __init__(self, min_distance: float = 20.0, max_distance: float = 200.0):
        """
        Initialize the mapper.
        
        Args:
            min_distance: Minimum distance (pixels) - maps to brightness 1
            max_distance: Maximum distance (pixels) - maps to brightness 254
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        
    def distance_to_brightness(self, distance: float) -> int:
        """
        Map distance to brightness value.
        
        Args:
            distance: Distance in pixels between thumb and index finger
            
        Returns:
            Brightness value (1-254)
        """
        # Clamp distance to min/max range
        distance = max(self.min_distance, min(self.max_distance, distance))
        
        # Normalize to 0-1 range
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        
        # Map to brightness range (1-254)
        brightness = int(normalized * 253) + 1
        
        return brightness


def list_available_lights(bridge_ip: str="192.168.1.2"):
    """List all available lights with their IDs and names."""
    try:
        b = Bridge(bridge_ip)
        try:
            b.get_api()
        except Exception:
            print("Please press the button on your Hue Bridge now...")
            print("Waiting 10 seconds...")
            import time
            time.sleep(10)
            b.connect()
        
        lights = b.get_light()
        print("\n" + "="*60)
        print("Available Lights:")
        print("="*60)
        for light_id, light_info in lights.items():
            print(f"  ID: {light_id:3} | Name: {light_info['name']}")
        print("="*60 + "\n")
        return lights
    except Exception as e:
        print(f"Error listing lights: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Control Philips Hue lights with hand gestures'
    )
    parser.add_argument(
        '--bridge-ip',
        type=str,
        required=True,
        default='192.168.1.2',
        help='IP address of your Philips Hue Bridge'
    )
    parser.add_argument(
        '--lights',
        type=str,  # Changed from int to str to accept names
        nargs='+',
        default=['1'],
        help='Light IDs or names to control (e.g., --lights 1 2 3 or --lights "Living Room" Bedroom)'
    )
    parser.add_argument(
        '--list-lights',
        action='store_true',
        help='List all available lights and exit'
    )
    parser.add_argument(
        '--min-distance',
        type=float,
        default=20.0,
        help='Minimum distance (pixels) for minimum brightness (default: 20)'
    )
    parser.add_argument(
        '--max-distance',
        type=float,
        default=200.0,
        help='Maximum distance (pixels) for maximum brightness (default: 200)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    
    args = parser.parse_args()
    
    # List lights if requested
    if args.list_lights:
        list_available_lights(args.bridge_ip)
        return
    
    # Initialize components
    print("Initializing hand tracker...")
    tracker = HandTracker()
    
    print(f"Connecting to Hue Bridge at {args.bridge_ip}...")
    hue_controller = HueController(args.bridge_ip, args.lights)
    
    print("Initializing brightness mapper...")
    mapper = GestureBrightnessMapper(args.min_distance, args.max_distance)
    
    # Initialize camera
    print(f"Opening camera {args.camera}...")
    
    # Use appropriate backend for the platform
    if sys.platform.startswith('linux'):
        # On Linux/WSL, use V4L2 backend explicitly for better USB passthrough support
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    elif sys.platform.startswith('win'):
        # On Windows, use DirectShow for better USB camera support
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Configure camera for USB passthrough (important for WSL)
    if sys.platform.startswith('linux'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid stale frames
        # Set a reasonable resolution (some cameras need this)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera resolution: {width}x{height} @ {fps:.2f} FPS")
    
    # Warm-up: Read a few frames to initialize the camera stream
    print("Initializing camera stream...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1}/5: OK")
            break
        time.sleep(0.2)
    else:
        print("  Warning: Could not read initial frames, continuing anyway...")
    
    print("\n" + "="*60)
    print("Hue Hand Control is running!")
    print("="*60)
    print("Controls:")
    print("  - Move your thumb and index finger closer/farther to control brightness")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset calibration (recalculate min/max distances)")
    print("="*60 + "\n")
    
    # For dynamic calibration
    observed_distances = []
    frame_timeout_count = 0
    max_timeout_count = 10  # Allow some timeouts before giving up
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_timeout_count += 1
                if frame_timeout_count > max_timeout_count:
                    print("Error: Could not read frame from camera after multiple attempts")
                    break
                # Try to reinitialize the camera
                if frame_timeout_count == 5:
                    print("Warning: Frame read timeout, attempting to reinitialize camera...")
                    cap.release()
                    time.sleep(0.5)
                    if sys.platform.startswith('linux'):
                        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    elif sys.platform.startswith('win'):
                        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(args.camera)
                time.sleep(0.1)
                continue
            
            # Reset timeout counter on successful read
            frame_timeout_count = 0
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            results = tracker.detect_landmarks(frame)
            
            if results.multi_hand_landmarks:
                # Get the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                tracker.draw_landmarks(frame, hand_landmarks)
                
                # Get thumb and index finger tips
                tips = tracker.get_thumb_index_tips(hand_landmarks, frame.shape)
                
                if tips:
                    thumb_pos, index_pos = tips
                    
                    # Draw fingertip positions
                    cv2.circle(frame, thumb_pos, 10, (0, 255, 0), -1)
                    cv2.circle(frame, index_pos, 10, (0, 255, 0), -1)
                    
                    # Draw line between fingertips
                    cv2.line(frame, thumb_pos, index_pos, (255, 0, 0), 2)
                    
                    # Calculate distance
                    distance = tracker.calculate_distance(thumb_pos, index_pos)
                    observed_distances.append(distance)
                    
                    # Map distance to brightness
                    brightness = mapper.distance_to_brightness(distance)
                    
                    # Update Hue lights
                    hue_controller.set_brightness(brightness)
                    
                    # Display information on frame
                    cv2.putText(frame, f"Distance: {distance:.1f}px", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Brightness: {brightness}/254", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display current light names/IDs
                    light_display = []
                    for light_id in args.lights:
                        display_name = hue_controller.get_light_display_name(light_id)
                        light_display.append(display_name)
                    lights_str = ", ".join(light_display)
                    cv2.putText(frame, f"Lights: {lights_str}", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Fingertips not detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Hue Hand Control', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset calibration
                if observed_distances:
                    new_min = min(observed_distances)
                    new_max = max(observed_distances)
                    mapper = GestureBrightnessMapper(new_min, new_max)
                    print(f"\nCalibration reset: min={new_min:.1f}px, max={new_max:.1f}px\n")
                    observed_distances = []
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nHue Hand Control stopped. Goodbye!")


if __name__ == "__main__":
    main()

