"""
Orchestrator Light Control - Point at lights to turn them green.
Uses hand tracking for pointing direction and pose tracking for user position.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import sys
import time
import threading
import json
from typing import Optional, Tuple, Dict, List
from phue import Bridge, Light
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BRIDGE_IP = "192.168.1.2"
CALIBRATION_FILE = "calibration_data.json"
POINTING_ANGLE_THRESHOLD = 60.0  # degrees - robust but responsive
GREEN_HUE = 25500  # Hue value for green
GREEN_SAT = 254  # Full saturation


class HandTracker:
    """Enhanced hand tracking with pointing vector calculation."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support both hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_landmarks(self, frame):
        """Detect hand landmarks in the frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
    
    def get_pointing_vector(self, landmarks, frame_shape) -> Optional[np.ndarray]:
        """
        Get 3D pointing direction vector from index finger.
        Returns normalized 3D vector in camera space, or None if not detected.
        """
        if not landmarks:
            return None
        
        # Get index finger tip and MCP (base of index finger)
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Calculate 3D direction vector (MediaPipe provides x, y, z in normalized coordinates)
        # z is depth estimate (smaller = closer)
        direction = np.array([
            index_tip.x - index_mcp.x,
            index_tip.y - index_mcp.y,
            index_tip.z - index_mcp.z
        ])
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        
        return direction / norm
    
    def get_hand_position(self, landmarks) -> Optional[np.ndarray]:
        """Get hand center position in 3D camera space."""
        if not landmarks:
            return None
        
        # Use wrist as hand position reference
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        return np.array([wrist.x, wrist.y, wrist.z])
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on the frame."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS
            )


class PoseTracker:
    """Pose tracking for user position estimation."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_landmarks(self, frame):
        """Detect pose landmarks in the frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results
    
    def get_user_center(self, landmarks) -> Optional[np.ndarray]:
        """
        Get user's center position relative to camera.
        Uses average of shoulders and hips for stability.
        """
        if not landmarks:
            return None
        
        try:
            # Get key body points
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate center (average of shoulders and hips)
            center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4.0
            center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4.0
            center_z = (left_shoulder.z + right_shoulder.z + left_hip.z + right_hip.z) / 4.0
            
            return np.array([center_x, center_y, center_z])
        except:
            return None
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on the frame."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS
            )


class LightPositionManager:
    """Manages light positions from room data."""
    
    def __init__(self, room_data: Dict):
        """
        Initialize with room data from fetch_light_positions.py output.
        
        Args:
            room_data: Dictionary with 'room' and 'lights' keys
        """
        self.positions = {}
        self.light_names = {}
        self.light_ids = []
        
        # Load positions from room data
        if 'room' in room_data and 'positions' in room_data['room']:
            self.positions = room_data['room']['positions']
        
        # Load light names
        if 'lights' in room_data:
            for light in room_data['lights']:
                light_id = str(light['id'])
                self.light_names[light_id] = light['name']
                if light_id not in self.positions:
                    # If no position, use default (0, 0, 0)
                    self.positions[light_id] = [0.0, 0.0, 0.0]
        
        self.light_ids = list(self.positions.keys())
    
    def get_light_position(self, light_id: str) -> Optional[np.ndarray]:
        """Get 3D position of a light."""
        if light_id in self.positions:
            return np.array(self.positions[light_id])
        return None
    
    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get all light positions as numpy arrays."""
        return {lid: np.array(pos) for lid, pos in self.positions.items()}
    
    def get_light_name(self, light_id: str) -> str:
        """Get light name by ID."""
        return self.light_names.get(light_id, f"Light {light_id}")


class CalibrationManager:
    """Manages camera calibration (position and orientation)."""
    
    def __init__(self, calibration_file: str = CALIBRATION_FILE):
        self.calibration_file = calibration_file
        self.camera_position = None
        self.camera_orientation = None
    
    def load_calibration(self) -> bool:
        """Load calibration from file."""
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                self.camera_position = np.array(data['camera_position'])
                self.camera_orientation = np.array(data['camera_orientation'])
                # Normalize orientation
                norm = np.linalg.norm(self.camera_orientation)
                if norm > 1e-6:
                    self.camera_orientation = self.camera_orientation / norm
                return True
        except:
            return False
    
    def save_calibration(self):
        """Save calibration to file."""
        data = {
            'camera_position': self.camera_position.tolist(),
            'camera_orientation': self.camera_orientation.tolist()
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calibrate_interactive(self, light_positions: Dict[str, np.ndarray], light_names: Dict[str, str]):
        """Interactive calibration: manual position entry, click-based orientation."""
        print("\n" + "="*60)
        print("CALIBRATION MODE")
        print("="*60)
        print("Step 1: Camera Position (Manual Entry)")
        print("  Enter the camera's position in the room (X, Y, Z coordinates).")
        print("  You can reference the 3D plot of lights to estimate coordinates.")
        print("="*60)
        
        # Show 3D plot of lights for reference
        fig_ref = plt.figure(figsize=(12, 10))
        ax_ref = fig_ref.add_subplot(111, projection='3d')
        
        # Plot lights
        for light_id, pos in light_positions.items():
            name = light_names.get(light_id, f"Light {light_id}")
            ax_ref.scatter(pos[0], pos[1], pos[2], s=200, alpha=0.7, c='blue', edgecolors='black')
            ax_ref.text(pos[0], pos[1], pos[2], f"  {light_id}\n  {name[:15]}", fontsize=7)
        
        # Set equal aspect ratio
        all_positions = np.array(list(light_positions.values()))
        if len(all_positions) > 0:
            max_range = np.array([
                all_positions[:, 0].max() - all_positions[:, 0].min(),
                all_positions[:, 1].max() - all_positions[:, 1].min(),
                all_positions[:, 2].max() - all_positions[:, 2].min()
            ]).max() / 2.0
            mid = all_positions.mean(axis=0)
            ax_ref.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax_ref.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax_ref.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax_ref.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax_ref.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax_ref.set_zlabel('Z Position', fontsize=12, fontweight='bold')
        ax_ref.set_title('Light Positions - Reference for Camera Position', fontsize=12, fontweight='bold')
        plt.show(block=False)
        
        # Get camera position manually
        while True:
            try:
                pos_input = input("\nEnter camera position (X Y Z) or 'plot' to see plot again: ").strip()
                if pos_input.lower() == 'plot':
                    plt.show(block=False)
                    continue
                
                coords = [float(x) for x in pos_input.split()]
                if len(coords) == 3:
                    self.camera_position = np.array(coords)
                    break
                else:
                    print("Please enter 3 coordinates (X Y Z)")
            except ValueError:
                print("Invalid input. Please enter numbers.")
            except KeyboardInterrupt:
                plt.close(fig_ref)
                return False
        
        plt.close(fig_ref)
        print(f"\n✓ Camera position set to: [{self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f}]")
        
        # Step 2: Manual orientation point entry
        print("\n" + "="*60)
        print("Step 2: Camera Orientation (Manual Entry)")
        print("  Enter a point in 3D space that the camera should point towards.")
        print("  The orientation will be calculated as the direction from camera position to this point.")
        print("  You can reference the 3D plot of lights to estimate coordinates.")
        print("="*60)
        
        # Show 3D plot with camera position for reference
        fig_orient = plt.figure(figsize=(12, 10))
        ax_orient = fig_orient.add_subplot(111, projection='3d')
        
        # Plot lights
        for light_id, pos in light_positions.items():
            name = light_names.get(light_id, f"Light {light_id}")
            ax_orient.scatter(pos[0], pos[1], pos[2], s=200, alpha=0.7, c='blue', edgecolors='black')
            ax_orient.text(pos[0], pos[1], pos[2], f"  {light_id}\n  {name[:15]}", fontsize=7)
        
        # Plot camera position
        ax_orient.scatter(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                         s=400, c='red', marker='^', label='Camera Position', edgecolors='black', linewidth=2)
        
        # Set equal aspect ratio
        all_positions = np.array(list(light_positions.values()))
        if len(all_positions) > 0:
            all_positions = np.vstack([all_positions, self.camera_position.reshape(1, -1)])
            max_range = np.array([
                all_positions[:, 0].max() - all_positions[:, 0].min(),
                all_positions[:, 1].max() - all_positions[:, 1].min(),
                all_positions[:, 2].max() - all_positions[:, 2].min()
            ]).max() / 2.0
            mid = all_positions.mean(axis=0)
            ax_orient.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax_orient.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax_orient.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax_orient.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax_orient.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax_orient.set_zlabel('Z Position', fontsize=12, fontweight='bold')
        ax_orient.set_title('Camera Position - Reference for Orientation Point', fontsize=12, fontweight='bold')
        ax_orient.legend()
        plt.show(block=False)
        
        # Get orientation point manually
        while True:
            try:
                point_input = input("\nEnter orientation point (X Y Z) that camera should point towards, or 'plot' to see plot again: ").strip()
                if point_input.lower() == 'plot':
                    plt.show(block=False)
                    continue
                
                coords = [float(x) for x in point_input.split()]
                if len(coords) == 3:
                    orientation_point = np.array(coords)
                    
                    # Calculate direction vector from camera to orientation point
                    direction = orientation_point - self.camera_position
                    norm = np.linalg.norm(direction)
                    
                    if norm > 1e-6:
                        self.camera_orientation = direction / norm
                        break
                    else:
                        print("Orientation point is too close to camera position. Please choose a different point.")
                else:
                    print("Please enter 3 coordinates (X Y Z)")
            except ValueError:
                print("Invalid input. Please enter numbers.")
            except KeyboardInterrupt:
                plt.close(fig_orient)
                return False
        
        plt.close(fig_orient)
        print(f"\n✓ Orientation point set to: [{orientation_point[0]:.2f}, {orientation_point[1]:.2f}, {orientation_point[2]:.2f}]")
        print(f"✓ Camera orientation calculated: [{self.camera_orientation[0]:.2f}, {self.camera_orientation[1]:.2f}, {self.camera_orientation[2]:.2f}]")
        
        # Show final calibration result
        print("\n" + "="*60)
        print("CALIBRATION RESULT")
        print("="*60)
        print(f"Camera Position: [{self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f}]")
        print(f"Orientation Point: [{orientation_point[0]:.2f}, {orientation_point[1]:.2f}, {orientation_point[2]:.2f}]")
        print(f"Camera Orientation: [{self.camera_orientation[0]:.2f}, {self.camera_orientation[1]:.2f}, {self.camera_orientation[2]:.2f}]")
        print("="*60)
        
        # Show final plot with calibration (non-blocking)
        fig_final = plt.figure(figsize=(12, 10))
        ax_final = fig_final.add_subplot(111, projection='3d')
        
        # Plot lights
        for light_id, pos in light_positions.items():
            name = light_names.get(light_id, f"Light {light_id}")
            ax_final.scatter(pos[0], pos[1], pos[2], s=200, alpha=0.7, c='blue', edgecolors='black')
            ax_final.text(pos[0], pos[1], pos[2], f"  {light_id}\n  {name[:15]}", fontsize=7)
        
        # Plot camera position
        ax_final.scatter(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                        s=400, c='red', marker='^', label='Camera Position', edgecolors='black', linewidth=2)
        
        # Plot orientation point
        ax_final.scatter(orientation_point[0], orientation_point[1], orientation_point[2],
                        s=300, c='orange', marker='o', label='Orientation Point', edgecolors='black', linewidth=2)
        
        # Plot camera orientation arrow
        direction_vec = orientation_point - self.camera_position
        ax_final.quiver(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                       direction_vec[0], direction_vec[1], direction_vec[2],
                       color='red', arrow_length_ratio=0.15, linewidth=3, label='Camera Direction')
        
        # Set limits
        all_positions = np.array(list(light_positions.values()))
        if len(all_positions) > 0:
            all_positions = np.vstack([all_positions, self.camera_position.reshape(1, -1), orientation_point.reshape(1, -1)])
            max_range = np.array([
                all_positions[:, 0].max() - all_positions[:, 0].min(),
                all_positions[:, 1].max() - all_positions[:, 1].min(),
                all_positions[:, 2].max() - all_positions[:, 2].min()
            ]).max() / 2.0
            mid = all_positions.mean(axis=0)
            ax_final.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax_final.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax_final.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax_final.set_xlabel('X')
        ax_final.set_ylabel('Y')
        ax_final.set_zlabel('Z')
        ax_final.set_title('Final Calibration Result')
        ax_final.legend()
        plt.show(block=False)  # Non-blocking so we can continue
        
        # Wait for user to press Enter to continue
        input("\nPress Enter to continue to save calibration...")
        
        # Save calibration
        save = input("\nSave this calibration? (y/n): ").strip().lower()
        if save == 'y':
            self.save_calibration()
            print("Calibration saved!")
            plt.close(fig_final)
            return True
        else:
            print("Calibration not saved.")
            plt.close(fig_final)
            return False


class CoordinateMapper:
    """Maps camera space to room 3D coordinates."""
    
    def __init__(self, camera_position: np.ndarray, camera_orientation: np.ndarray):
        """
        Initialize coordinate mapper.
        
        Args:
            camera_position: Camera position in room coordinates [X, Y, Z]
            camera_orientation: Camera forward direction (normalized) [X, Y, Z]
        """
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        
        # Create rotation matrix to align camera space with room space
        # Camera's forward is +Z in camera space, we need to align with camera_orientation
        # This is a simplified approach - assumes camera's up is +Y
        forward = camera_orientation
        # Create a right vector (assume up is roughly +Y in room space)
        up_room = np.array([0, 1, 0])
        right = np.cross(forward, up_room)
        if np.linalg.norm(right) < 1e-6:
            # If forward is parallel to up, use different up
            up_room = np.array([0, 0, 1])
            right = np.cross(forward, up_room)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Rotation matrix: columns are right, up, forward in room space
        self.rotation_matrix = np.column_stack([right, up, forward])
    
    def transform_to_room_coords(self, camera_vector: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Transform a vector from camera space to room space.
        
        Args:
            camera_vector: Vector in camera space (normalized MediaPipe coordinates)
            scale: Scale factor to convert normalized coordinates to room scale
        
        Returns:
            Vector in room coordinates
        """
        # Rotate vector to room orientation
        room_vector = self.rotation_matrix @ camera_vector
        
        # Scale (MediaPipe coordinates are normalized, need to scale to room dimensions)
        return room_vector * scale
    
    def get_user_position(self, pose_center: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Calculate user's 3D position in room coordinates.
        
        Args:
            pose_center: User center in camera space (normalized MediaPipe coordinates)
            scale: Scale factor to convert to room coordinates (meters or room units)
        
        Returns:
            User position in room coordinates
        """
        # Transform pose offset to room space
        # MediaPipe z is depth (smaller = closer), so we use it as distance
        # Convert normalized coordinates to room space
        offset = self.transform_to_room_coords(pose_center, scale)
        
        # User position = camera position + offset
        return self.camera_position + offset


class OrchestratorLightController:
    """Controls individual lights, storing and restoring their states."""
    
    def __init__(self, bridge: Bridge, light_ids: List[str]):
        """
        Initialize light controller.
        
        Args:
            bridge: Connected Hue Bridge
            light_ids: List of light IDs to control
        """
        self.bridge = bridge
        self.light_ids = light_ids
        self.lights = {}
        self.original_states = {}
        self.active_lights = set()
        self.update_lock = threading.Lock()
        self.last_update_time = 0
        self.min_update_interval = 0.05  # 50ms minimum between updates
        
        # Load light objects and store original states
        for light_id in light_ids:
            try:
                light = bridge[int(light_id)]
                self.lights[light_id] = light
                # Store original state
                self.original_states[light_id] = {
                    'on': light.on,
                    'hue': light.hue if hasattr(light, 'hue') else None,
                    'saturation': light.saturation if hasattr(light, 'saturation') else None,
                    'brightness': light.brightness if hasattr(light, 'brightness') else None,
                    'colormode': light.colormode if hasattr(light, 'colormode') else None
                }
            except Exception as e:
                print(f"Warning: Could not load light {light_id}: {e}")
    
    def set_light_green(self, light_id: str):
        """Set a light to green."""
        if light_id not in self.lights:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time < self.min_update_interval:
            return
        
        light = self.lights[light_id]
        try:
            if not light.on:
                light.on = True
            # Set to green, maintain current brightness
            if light.brightness == 0:
                light.brightness = 254
            light.hue = GREEN_HUE
            light.saturation = GREEN_SAT
            self.active_lights.add(light_id)
            self.last_update_time = current_time
        except Exception as e:
            print(f"Error setting light {light_id} to green: {e}")
    
    def restore_light_state(self, light_id: str):
        """Restore a light to its original state."""
        if light_id not in self.lights or light_id not in self.original_states:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time < self.min_update_interval:
            return
        
        light = self.lights[light_id]
        state = self.original_states[light_id]
        
        try:
            light.on = state['on']
            if state['hue'] is not None:
                light.hue = state['hue']
            if state['saturation'] is not None:
                light.saturation = state['saturation']
            if state['brightness'] is not None:
                light.brightness = state['brightness']
            self.active_lights.discard(light_id)
            self.last_update_time = current_time
        except Exception as e:
            print(f"Error restoring light {light_id}: {e}")
    
    def update_lights(self, active_light_ids: set):
        """Update lights based on which are active."""
        with self.update_lock:
            # Set active lights to green
            for light_id in active_light_ids:
                if light_id in self.lights:
                    self.set_light_green(light_id)
            
            # Restore inactive lights
            to_restore = self.active_lights - active_light_ids
            for light_id in to_restore:
                self.restore_light_state(light_id)


class Live3DVisualizer:
    """Live 3D visualization of lights, camera, and user position."""
    
    def __init__(self, light_positions: Dict[str, np.ndarray], light_names: Dict[str, str],
                 camera_position: np.ndarray, camera_orientation: np.ndarray):
        self.light_positions = light_positions
        self.light_names = light_names
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        
        self.fig = None
        self.ax = None
        self.running = False
        self.update_lock = threading.Lock()
        self.current_user_pos = None
        self.current_pointing_vector = None
        self.active_lights = set()
        
        # Start visualization
        self.start()
    
    def start(self):
        """Start the visualization window."""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.running = True
        
        # Initial plot
        self._update_plot()
    
    def _update_plot(self):
        """Update the 3D plot."""
        self.ax.clear()
        
        # Plot lights
        for light_id, pos in self.light_positions.items():
            name = self.light_names.get(light_id, f"Light {light_id}")
            is_active = light_id in self.active_lights
            
            color = 'green' if is_active else 'blue'
            size = 300 if is_active else 200
            self.ax.scatter(pos[0], pos[1], pos[2], s=size, c=color, alpha=0.7, edgecolors='black')
            self.ax.text(pos[0], pos[1], pos[2], f"  {light_id}\n  {name[:15]}", fontsize=7)
        
        # Plot camera position
        self.ax.scatter(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                       s=400, c='red', marker='^', label='Camera', edgecolors='black', linewidth=2)
        
        # Plot camera orientation
        arrow_length = 0.3
        arrow_end = self.camera_position + self.camera_orientation * arrow_length
        self.ax.quiver(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                      self.camera_orientation[0] * arrow_length,
                      self.camera_orientation[1] * arrow_length,
                      self.camera_orientation[2] * arrow_length,
                      color='red', arrow_length_ratio=0.3, linewidth=2)
        
        # Plot user position if available
        if self.current_user_pos is not None:
            self.ax.scatter(self.current_user_pos[0], self.current_user_pos[1], self.current_user_pos[2],
                           s=500, c='orange', marker='o', label='You', edgecolors='black', linewidth=2)
            
            # Plot pointing vector if available
            if self.current_pointing_vector is not None:
                arrow_length = 1.0
                arrow_end = self.current_user_pos + self.current_pointing_vector * arrow_length
                self.ax.quiver(self.current_user_pos[0], self.current_user_pos[1], self.current_user_pos[2],
                              self.current_pointing_vector[0] * arrow_length,
                              self.current_pointing_vector[1] * arrow_length,
                              self.current_pointing_vector[2] * arrow_length,
                              color='yellow', arrow_length_ratio=0.2, linewidth=3)
        
        self.ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('Z Position', fontsize=12, fontweight='bold')
        self.ax.set_title('Live 3D Map - Orchestrator Control', fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        all_positions = list(self.light_positions.values())
        if self.current_user_pos is not None:
            all_positions.append(self.current_user_pos)
        if all_positions:
            positions_array = np.array(all_positions)
            max_range = np.array([
                positions_array[:, 0].max() - positions_array[:, 0].min(),
                positions_array[:, 1].max() - positions_array[:, 1].min(),
                positions_array[:, 2].max() - positions_array[:, 2].min()
            ]).max() / 2.0
            
            mid = positions_array.mean(axis=0)
            self.ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            self.ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            self.ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)
    
    def update(self, user_pos: Optional[np.ndarray], pointing_vector: Optional[np.ndarray],
               active_lights: set):
        """Update visualization with current state."""
        with self.update_lock:
            self.current_user_pos = user_pos
            self.current_pointing_vector = pointing_vector
            self.active_lights = active_lights.copy()
            self._update_plot()
    
    def stop(self):
        """Stop the visualization."""
        self.running = False
        plt.ioff()
        if self.fig:
            plt.close(self.fig)


def find_lights_in_cone(user_pos: np.ndarray, pointing_vector: np.ndarray,
                        light_positions: Dict[str, np.ndarray],
                        angle_threshold: float = POINTING_ANGLE_THRESHOLD) -> set:
    """
    Find lights within the pointing cone.
    
    Args:
        user_pos: User's 3D position
        pointing_vector: Normalized pointing direction vector
        light_positions: Dictionary of light_id -> position
        angle_threshold: Maximum angle in degrees
    
    Returns:
        Set of light IDs being pointed at
    """
    active_lights = set()
    threshold_cos = math.cos(math.radians(angle_threshold))
    
    for light_id, light_pos in light_positions.items():
        # Vector from user to light
        to_light = light_pos - user_pos
        distance = np.linalg.norm(to_light)
        
        if distance < 1e-6:  # Light is at user position (shouldn't happen)
            continue
        
        # Normalize
        to_light_normalized = to_light / distance
        
        # Calculate angle between pointing vector and light direction
        cos_angle = np.dot(pointing_vector, to_light_normalized)
        
        # Check if within cone
        if cos_angle >= threshold_cos:
            active_lights.add(light_id)
    
    return active_lights


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


def get_user_position_manual(light_positions: Dict[str, np.ndarray], 
                             light_names: Dict[str, str],
                             camera_position: np.ndarray) -> Optional[np.ndarray]:
    """
    Get user position manually with 3D plot reference.
    
    Args:
        light_positions: Dictionary of light positions
        light_names: Dictionary of light names
        camera_position: Camera position for reference
    
    Returns:
        User position as numpy array, or None if cancelled
    """
    print("\n" + "="*60)
    print("USER POSITION SETUP")
    print("="*60)
    print("Enter your position in the room (X, Y, Z coordinates).")
    print("You can reference the 3D plot of lights to estimate coordinates.")
    print("="*60)
    
    # Show 3D plot of lights for reference
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot lights
    for light_id, pos in light_positions.items():
        name = light_names.get(light_id, f"Light {light_id}")
        ax.scatter(pos[0], pos[1], pos[2], s=200, alpha=0.7, c='blue', edgecolors='black')
        ax.text(pos[0], pos[1], pos[2], f"  {light_id}\n  {name[:15]}", fontsize=7)
    
    # Plot camera position
    ax.scatter(camera_position[0], camera_position[1], camera_position[2],
              s=400, c='red', marker='^', label='Camera Position', edgecolors='black', linewidth=2)
    
    # Set equal aspect ratio
    all_positions = np.array(list(light_positions.values()))
    if len(all_positions) > 0:
        all_positions = np.vstack([all_positions, camera_position.reshape(1, -1)])
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        mid = all_positions.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position', fontsize=12, fontweight='bold')
    ax.set_title('Light Positions - Reference for Your Position', fontsize=12, fontweight='bold')
    ax.legend()
    plt.show(block=False)
    
    # Get user position manually
    while True:
        try:
            pos_input = input("\nEnter your position (X Y Z) or 'plot' to see plot again: ").strip()
            if pos_input.lower() == 'plot':
                plt.show(block=False)
                continue
            
            coords = [float(x) for x in pos_input.split()]
            if len(coords) == 3:
                user_pos = np.array(coords)
                plt.close(fig)
                print(f"\n✓ Your position set to: [{user_pos[0]:.2f}, {user_pos[1]:.2f}, {user_pos[2]:.2f}]")
                return user_pos
            else:
                print("Please enter 3 coordinates (X Y Z)")
        except ValueError:
            print("Invalid input. Please enter numbers.")
        except KeyboardInterrupt:
            plt.close(fig)
            print("\nCancelled.")
            return None


def load_room_data(filename: str) -> Optional[Dict]:
    """Load room data from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None


def main():
    print("=" * 60)
    print("Orchestrator Light Control")
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
    
    # Try to load room data from JSON file
    room_data_filename = f"hue_room_data_{selected_room['name'].replace(' ', '_')}.json"
    room_data = load_room_data(room_data_filename)
    
    if not room_data:
        print(f"\n⚠ Could not load room data from {room_data_filename}")
        print("Please run fetch_light_positions.py first to generate room data.")
        return
    
    # Initialize light position manager
    light_manager = LightPositionManager(room_data)
    light_positions = light_manager.get_all_positions()
    
    if not light_positions:
        print("\n⚠ No light positions found in room data.")
        print("Make sure the room has Entertainment Area positions set up.")
        return
    
    print(f"\n✓ Loaded {len(light_positions)} light positions")
    
    # Calibration
    calib_manager = CalibrationManager()
    if not calib_manager.load_calibration():
        print("\nNo calibration found. Starting calibration...")
        if not calib_manager.calibrate_interactive(light_positions, light_manager.light_names):
            print("Calibration cancelled.")
            return
    else:
        print("\n✓ Loaded existing calibration")
        print(f"  Camera position: {calib_manager.camera_position}")
        print(f"  Camera orientation: {calib_manager.camera_orientation}")
        
        recalibrate = input("\nRecalibrate? (y/n): ").strip().lower()
        if recalibrate == 'y':
            if not calib_manager.calibrate_interactive(light_positions, light_manager.light_names):
                print("Calibration cancelled.")
                return
    
    # Initialize coordinate mapper
    coord_mapper = CoordinateMapper(calib_manager.camera_position, calib_manager.camera_orientation)
    
    # Get user position manually
    print("\n" + "="*60)
    print("USER POSITION SETUP")
    print("="*60)
    user_position = get_user_position_manual(light_positions, light_manager.light_names,
                                             calib_manager.camera_position)
    
    if user_position is None:
        print("User position setup cancelled.")
        return
    
    # Initialize light controller
    light_controller = OrchestratorLightController(bridge, light_manager.light_ids)
    
    # Initialize trackers
    hand_tracker = HandTracker()
    pose_tracker = PoseTracker()
    
    # Initialize live 3D visualizer
    visualizer = Live3DVisualizer(light_positions, light_manager.light_names,
                                  calib_manager.camera_position, calib_manager.camera_orientation)
    
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
        visualizer.stop()
        return
    
    # Configure camera for low latency
    if sys.platform.startswith('linux'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    elif sys.platform.startswith('win'):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Warm-up camera
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.2)
    
    print("\n" + "=" * 60)
    print("Orchestrator Control Active!")
    print("=" * 60)
    print("Controls:")
    print("  - Point at lights to turn them green")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to recalibrate camera")
    print("  - Press 'p' to update your position")
    print("=" * 60 + "\n")
    
    visualization_update_interval = 0.1  # Update 3D plot every 100ms
    last_viz_update = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand and pose landmarks
            hand_results = hand_tracker.detect_landmarks(frame)
            pose_results = pose_tracker.detect_landmarks(frame)
            
            # Use manually entered position (not inferred from pose)
            user_pos = user_position.copy() if user_position is not None else None
            pointing_vector = None
            active_lights = set()
            
            # Draw pose landmarks for visualization (but don't use for position calculation)
            if pose_results.pose_landmarks:
                pose_tracker.draw_landmarks(frame, pose_results.pose_landmarks)
            
            # Get pointing vector from hand
            if hand_results.multi_hand_landmarks:
                # Use first detected hand
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                camera_pointing = hand_tracker.get_pointing_vector(hand_landmarks, frame.shape)
                
                if camera_pointing is not None and user_pos is not None:
                    # Transform pointing vector to room coordinates
                    pointing_vector = coord_mapper.transform_to_room_coords(camera_pointing, scale=1.0)
                    pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)  # Normalize
                    
                    # Find lights in pointing cone
                    active_lights = find_lights_in_cone(user_pos, pointing_vector, light_positions)
                    
                    # Update lights
                    light_controller.update_lights(active_lights)
                
                hand_tracker.draw_landmarks(frame, hand_landmarks)
            
            # Update visualization periodically
            current_time = time.time()
            if current_time - last_viz_update >= visualization_update_interval:
                visualizer.update(user_pos, pointing_vector, active_lights)
                last_viz_update = current_time
            
            # Draw status on frame
            if user_pos is not None:
                cv2.putText(frame, f"User Pos (Manual): [{user_pos[0]:.2f}, {user_pos[1]:.2f}, {user_pos[2]:.2f}]",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Pose tracking: Visualization only",
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if pointing_vector is not None:
                cv2.putText(frame, f"Pointing: [{pointing_vector[0]:.2f}, {pointing_vector[1]:.2f}, {pointing_vector[2]:.2f}]",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            active_count = len(active_lights)
            cv2.putText(frame, f"Active Lights: {active_count}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if active_lights:
                active_names = [light_manager.get_light_name(lid) for lid in active_lights]
                cv2.putText(frame, f"Lights: {', '.join(active_names[:3])}",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display frame
            cv2.imshow('Orchestrator Light Control', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Recalibrate camera
                if calib_manager.calibrate_interactive(light_positions, light_manager.light_names):
                    coord_mapper = CoordinateMapper(calib_manager.camera_position, calib_manager.camera_orientation)
                    visualizer = Live3DVisualizer(light_positions, light_manager.light_names,
                                                  calib_manager.camera_position, calib_manager.camera_orientation)
            elif key == ord('p'):
                # Update user position
                print("\nUpdating user position...")
                new_pos = get_user_position_manual(light_positions, light_manager.light_names,
                                                   calib_manager.camera_position)
                if new_pos is not None:
                    user_position = new_pos
                    print(f"✓ Position updated to: [{user_position[0]:.2f}, {user_position[1]:.2f}, {user_position[2]:.2f}]")
                else:
                    print("Position update cancelled.")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Restore all lights
        print("\nRestoring all lights to original state...")
        for light_id in light_controller.light_ids:
            light_controller.restore_light_state(light_id)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        visualizer.stop()
        print("\nOrchestrator control stopped. Goodbye!")


if __name__ == "__main__":
    main()

