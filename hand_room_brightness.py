"""
Control room brightness using hand gestures - thumb and index finger distance.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import sys
import time
import threading
from typing import Optional, Tuple
from phue import Bridge, Group

BRIDGE_IP = "192.168.1.2"


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


class RoomBrightnessController:
    """Controller for room brightness using Hue Groups."""
    
    def __init__(self, bridge: Bridge, group_id: str):
        """
        Initialize room brightness controller.
        
        Args:
            bridge: Connected Hue Bridge
            group_id: Group/room ID to control
        """
        self.bridge = bridge
        self.group_id = group_id
        self.group = Group(bridge, group_id)
        self.last_brightness = None
        self.last_update_time = 0
        self.update_lock = threading.Lock()
        
    def _update_brightness_thread(self, brightness: int):
        """Thread function to update brightness without blocking."""
        try:
            # Turn on the group first (only if needed)
            if not self.group.on:
                self.group.on = True
            # Set brightness for the entire group
            self.group.brightness = brightness
        except Exception as e:
            print(f"Error setting brightness: {e}")
    
    def set_brightness(self, brightness: int, min_update_interval: float = 0.03):
        """
        Set brightness for all lights in the room (non-blocking).
        
        Args:
            brightness: Brightness value (1-254)
            min_update_interval: Minimum time between updates (reduced for responsiveness)
        """
        # Clamp brightness to valid range
        brightness = max(1, min(254, int(brightness)))
        
        # Rate limiting - more aggressive for responsiveness
        current_time = time.time()
        if (self.last_brightness == brightness and 
            current_time - self.last_update_time < min_update_interval):
            return brightness
        
        # Update if brightness changed (reduced threshold for more sensitivity)
        if self.last_brightness is not None and abs(self.last_brightness - brightness) < 1:
            return brightness
        
        # Use lock to prevent race conditions
        with self.update_lock:
            self.last_brightness = brightness
            self.last_update_time = current_time
        
        # Update brightness in a separate thread to avoid blocking
        thread = threading.Thread(target=self._update_brightness_thread, args=(brightness,))
        thread.daemon = True
        thread.start()
        
        return brightness


def get_rooms(bridge):
    """Get all available rooms/groups from the bridge."""
    try:
        groups = bridge.get_group()
        rooms = []
        for group_id, group_info in groups.items():
            if group_id != '0':
                rooms.append({
                    'id': group_id,
                    'name': group_info.get('name', f'Group {group_id}'),
                    'lights': group_info.get('lights', [])
                })
        return rooms
    except Exception as e:
        print(f"Error getting rooms: {e}")
        return []


def display_rooms(rooms):
    """Display available rooms."""
    if not rooms:
        print("No rooms found!")
        return
    
    print("\nAvailable Rooms:")
    print("=" * 60)
    for i, room in enumerate(rooms, 1):
        num_lights = len(room['lights'])
        print(f"  {i}. {room['name']} (ID: {room['id']}, {num_lights} light(s))")
    print("=" * 60)


def select_room(rooms):
    """Let user select a room."""
    if not rooms:
        return None
    
    while True:
        try:
            choice = input(f"\nSelect a room (1-{len(rooms)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(rooms):
                return rooms[index]
            else:
                print(f"Please enter a number between 1 and {len(rooms)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def main():
    print("=" * 60)
    print("Hand Gesture Room Brightness Control")
    print("=" * 60)
    
    # Connect to bridge
    print(f"\nConnecting to Hue Bridge at {BRIDGE_IP}...")
    try:
        bridge = Bridge(BRIDGE_IP)
        bridge.get_api()
        print("✓ Connected to bridge")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("Press the button on your Hue Bridge and try again...")
        return
    
    # Get and select room
    print("\nFetching rooms...")
    rooms = get_rooms(bridge)
    
    if not rooms:
        print("No rooms found. Make sure you have rooms/groups set up in the Hue app.")
        return
    
    display_rooms(rooms)
    selected_room = select_room(rooms)
    
    if not selected_room:
        return
    
    # Initialize room controller
    print(f"\nInitializing brightness control for '{selected_room['name']}'...")
    room_controller = RoomBrightnessController(bridge, selected_room['id'])
    
    # Initialize hand tracker
    print("Initializing hand tracker...")
    tracker = HandTracker()
    
    # Initialize brightness mapper
    print("Initializing brightness mapper...")
    mapper = GestureBrightnessMapper(min_distance=20.0, max_distance=200.0)
    
    # Initialize camera
    print(f"\nOpening camera...")
    if sys.platform.startswith('linux'):
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    elif sys.platform.startswith('win'):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Configure camera for low latency
    if sys.platform.startswith('linux'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    elif sys.platform.startswith('win'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
    
    # Warm-up camera
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.2)
    
    print("\n" + "=" * 60)
    print("Hand Gesture Control Active!")
    print("=" * 60)
    print("Controls:")
    print("  - Move thumb and index finger closer/farther to control brightness")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset calibration (recalculate min/max distances)")
    print("=" * 60 + "\n")
    
    # For dynamic calibration
    observed_distances = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
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
                    
                    # Update room brightness
                    room_controller.set_brightness(brightness)
                    
                    # Display information on frame
                    cv2.putText(frame, f"Distance: {distance:.1f}px", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Brightness: {brightness}/254", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Room: {selected_room['name']}", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Fingertips not detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Hand Gesture Room Brightness Control', frame)
            
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
        print("\nHand gesture control stopped. Goodbye!")


if __name__ == "__main__":
    main()

