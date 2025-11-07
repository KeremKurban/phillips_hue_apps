"""
Hue Hand Control - Meta Quest 2 Version

This version is optimized for Meta Quest 2 camera input.
It tracks thumb and index finger tips using MediaPipe,
calculates the distance between them, and uses that distance
to control the brightness of Philips Hue lights.

Designed for use with Quest 2 passthrough camera or Quest Link/Air Link.
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

# Import shared classes from main.py
from main import HandTracker, HueController, GestureBrightnessMapper, list_available_lights


def detect_quest2_camera():
    """
    Attempt to detect Quest 2 camera by checking available cameras.
    Returns the index of a likely Quest 2 camera, or None if not found.
    """
    print("Detecting available cameras...")
    candidates = []
    
    for i in range(10):  # Check up to 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                candidates.append({
                    'index': i,
                    'width': width,
                    'height': height
                })
            cap.release()
    
    if candidates:
        print(f"Found {len(candidates)} camera(s):")
        for cam in candidates:
            print(f"  Camera {cam['index']}: {cam['width']}x{cam['height']}")
        
        # Quest 2 typically has specific resolutions, but we'll just return the first non-zero camera
        # User can specify with --camera if needed
        if len(candidates) > 1:
            print("\nNote: Multiple cameras detected. Quest 2 is often camera 1 or 2.")
            print("Use --camera to specify which camera to use.")
            return candidates[1]['index'] if len(candidates) > 1 else candidates[0]['index']
        return candidates[0]['index']
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Control Philips Hue lights with hand gestures using Meta Quest 2 camera',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quest 2 Setup:
  1. Connect Quest 2 via Quest Link or Air Link
  2. Enable passthrough mode if available
  3. Ensure Quest 2 appears as a camera device on your PC
  4. Run this script with the appropriate camera index

Tips for Quest 2:
  - Use passthrough mode to see your hands in VR
  - Position your hands comfortably in front of the headset
  - Good lighting improves hand tracking accuracy
  - The Quest 2 camera may appear as camera index 1, 2, or higher
        """
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
        type=str,
        nargs='+',
        default=['4'],
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
        default=None,
        help='Camera index (auto-detects Quest 2 if not specified)'
    )
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List all available cameras and exit'
    )
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        import list_cameras
        list_cameras.list_cameras()
        return
    
    # List lights if requested
    if args.list_lights:
        list_available_lights(args.bridge_ip)
        return
    
    # Detect Quest 2 camera if not specified
    if args.camera is None:
        detected_camera = detect_quest2_camera()
        if detected_camera is None:
            print("Error: No cameras detected. Please specify --camera index or use --list-cameras to see available cameras.")
            sys.exit(1)
        args.camera = detected_camera
        print(f"Using detected camera: {args.camera}")
    
    # Initialize components
    print("\nInitializing hand tracker...")
    tracker = HandTracker()
    
    print(f"Connecting to Hue Bridge at {args.bridge_ip}...")
    hue_controller = HueController(args.bridge_ip, args.lights)
    
    print("Initializing brightness mapper...")
    mapper = GestureBrightnessMapper(args.min_distance, args.max_distance)
    
    # Initialize Quest 2 camera
    print(f"\nOpening Quest 2 camera (index {args.camera})...")
    
    # On Linux/WSL, use V4L2 backend explicitly for better USB passthrough support
    if sys.platform.startswith('linux'):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        print("\nTroubleshooting:")
        print("  1. Make sure Quest 2 is connected via Quest Link or Air Link")
        print("  2. Try running: python list_cameras.py to see available cameras")
        print("  3. Try different camera indices: --camera 0, --camera 1, --camera 2")
        sys.exit(1)
    
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
    print("Hue Hand Control - Quest 2 Mode")
    print("="*60)
    print("Controls:")
    print("  - Move your thumb and index finger closer/farther to control brightness")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset calibration (recalculate min/max distances)")
    print("\nQuest 2 Tips:")
    print("  - Use passthrough mode to see your hands")
    print("  - Position hands comfortably in front of the headset")
    print("  - Good lighting improves tracking accuracy")
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
                    print("Error: Could not read frame from Quest 2 camera after multiple attempts")
                    print("Make sure Quest 2 is still connected and Quest Link/Air Link is active.")
                    break
                # Try to reinitialize the camera
                if frame_timeout_count == 5:
                    print("Warning: Frame read timeout, attempting to reinitialize camera...")
                    cap.release()
                    time.sleep(0.5)
                    if sys.platform.startswith('linux'):
                        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    else:
                        cap = cv2.VideoCapture(args.camera)
                time.sleep(0.1)
                continue
            
            # Reset timeout counter on successful read
            frame_timeout_count = 0
            
            # Quest 2 passthrough may already be mirrored, but we'll flip it for consistency
            # Comment out if the image appears backwards
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            results = tracker.detect_landmarks(frame)
            
            if results.multi_hand_landmarks:
                # Get the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks with enhanced visibility for VR
                tracker.draw_landmarks(frame, hand_landmarks)
                
                # Get thumb and index finger tips
                tips = tracker.get_thumb_index_tips(hand_landmarks, frame.shape)
                
                if tips:
                    thumb_pos, index_pos = tips
                    
                    # Draw fingertip positions with larger circles for VR visibility
                    cv2.circle(frame, thumb_pos, 15, (0, 255, 0), -1)
                    cv2.circle(frame, index_pos, 15, (0, 255, 0), -1)
                    
                    # Draw thicker line between fingertips for VR visibility
                    cv2.line(frame, thumb_pos, index_pos, (255, 0, 0), 3)
                    
                    # Calculate distance
                    distance = tracker.calculate_distance(thumb_pos, index_pos)
                    observed_distances.append(distance)
                    
                    # Map distance to brightness
                    brightness = mapper.distance_to_brightness(distance)
                    
                    # Update Hue lights
                    hue_controller.set_brightness(brightness)
                    
                    # Display information on frame with larger text for VR
                    cv2.putText(frame, f"Distance: {distance:.1f}px", 
                              (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Brightness: {brightness}/254", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Display current light names/IDs
                    light_display = []
                    for light_id in args.lights:
                        display_name = hue_controller.get_light_display_name(light_id)
                        light_display.append(display_name)
                    lights_str = ", ".join(light_display)
                    cv2.putText(frame, f"Lights: {lights_str}", 
                              (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Add Quest 2 mode indicator
                    cv2.putText(frame, "Quest 2 Mode", 
                              (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "Fingertips not detected", 
                              (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "No hand detected - Show your hand to the Quest 2", 
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Hue Hand Control - Quest 2', frame)
            
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
        print("\nHue Hand Control (Quest 2) stopped. Goodbye!")


if __name__ == "__main__":
    main()

