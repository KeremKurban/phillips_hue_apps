"""
Light Calibration by Pointing - Calibrate light positions by pointing at them.
Turns each light red one by one, tracks your pointing direction for 5 seconds,
and calculates light positions based on your pointing.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import sys
import time
import json
import threading
from typing import Optional, Dict, List
from phue import Bridge, Light
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BRIDGE_IP = "192.168.1.2"
CALIBRATION_FILE = "calibration_data.json"
LIGHT_POSITIONS_FILE = "calibrated_light_positions.json"
USER_POSITION_FILE = "user_position.json"
RED_HUE = 0  # Hue value for red
RED_SAT = 254  # Full saturation
GREEN_HUE = 25500  # Hue value for green
GREEN_SAT = 254  # Full saturation
POINTING_ANGLE_THRESHOLD = 35.0  # degrees
CALIBRATION_DURATION = 5.0  # seconds to track pointing for each light


class HandTracker:
    """Enhanced hand tracking with pointing vector calculation."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
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
        
        # Calculate 3D direction vector
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
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on the frame."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS
            )


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
        print(f"\n✓ Camera orientation calculated: [{self.camera_orientation[0]:.2f}, {self.camera_orientation[1]:.2f}, {self.camera_orientation[2]:.2f}]")
        
        # Save calibration
        save = input("\nSave this calibration? (y/n): ").strip().lower()
        if save == 'y':
            self.save_calibration()
            print("Calibration saved!")
            return True
        else:
            print("Calibration not saved.")
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
        forward = camera_orientation
        up_room = np.array([0, 1, 0])
        right = np.cross(forward, up_room)
        if np.linalg.norm(right) < 1e-6:
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


class LightController:
    """Controls individual lights."""
    
    def __init__(self, bridge: Bridge, light_ids: List[str]):
        self.bridge = bridge
        self.light_ids = light_ids
        self.lights = {}
        self.original_states = {}
        
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
                }
            except Exception as e:
                print(f"Warning: Could not load light {light_id}: {e}")
    
    def set_light_red(self, light_id: str):
        """Set a light to red."""
        if light_id not in self.lights:
            return
        
        light = self.lights[light_id]
        try:
            if not light.on:
                light.on = True
            if light.brightness == 0:
                light.brightness = 254
            light.hue = RED_HUE
            light.saturation = RED_SAT
        except Exception as e:
            print(f"Error setting light {light_id} to red: {e}")
    
    def restore_light_state(self, light_id: str):
        """Restore a light to its original state."""
        if light_id not in self.lights or light_id not in self.original_states:
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
        except Exception as e:
            print(f"Error restoring light {light_id}: {e}")
    
    def restore_all(self):
        """Restore all lights to original state."""
        for light_id in self.light_ids:
            self.restore_light_state(light_id)


class OrchestratorLightController:
    """Controls individual lights, storing and restoring their states."""
    
    def __init__(self, bridge: Bridge, light_ids: List[str]):
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
            for light_id in active_light_ids:
                if light_id in self.lights:
                    self.set_light_green(light_id)
            
            to_restore = self.active_lights - active_light_ids
            for light_id in to_restore:
                self.restore_light_state(light_id)


def draw_cone_3d(ax, origin: np.ndarray, direction: np.ndarray, 
                 angle_degrees: float, length: float, 
                 color: str = 'yellow', alpha: float = 0.3, 
                 resolution: int = 20):
    """Draw a 3D cone in matplotlib."""
    direction = direction / np.linalg.norm(direction)
    angle_rad = math.radians(angle_degrees)
    radius = length * math.tan(angle_rad)
    
    if abs(direction[2]) < 0.9:
        perp = np.array([0, 0, 1])
    else:
        perp = np.array([1, 0, 0])
    
    u = np.cross(direction, perp)
    u = u / np.linalg.norm(u)
    v = np.cross(direction, u)
    v = v / np.linalg.norm(v)
    
    theta = np.linspace(0, 2 * np.pi, resolution)
    t = np.linspace(0, 1, 15)
    theta_mesh, t_mesh = np.meshgrid(theta, t)
    
    r = radius * (1 - t_mesh)
    x_circle = r * np.cos(theta_mesh)
    y_circle = r * np.sin(theta_mesh)
    
    x_circle_flat = x_circle.flatten()
    y_circle_flat = y_circle.flatten()
    t_flat = t_mesh.flatten()
    
    local_points = (x_circle_flat[:, np.newaxis] * u + 
                   y_circle_flat[:, np.newaxis] * v + 
                   t_flat[:, np.newaxis] * length * direction)
    world_points = origin + local_points
    
    x_world = world_points[:, 0].reshape(x_circle.shape)
    y_world = world_points[:, 1].reshape(x_circle.shape)
    z_world = world_points[:, 2].reshape(x_circle.shape)
    
    ax.plot_surface(x_world, y_world, z_world, color=color, alpha=alpha, 
                   linewidth=0, shade=True, antialiased=True)
    
    base_circle_points = radius * (np.cos(theta)[:, np.newaxis] * u + 
                                   np.sin(theta)[:, np.newaxis] * v) + length * direction
    base_circle_world = origin + base_circle_points
    
    ax.plot(base_circle_world[:, 0], base_circle_world[:, 1], base_circle_world[:, 2], 
            color=color, linewidth=2.5, alpha=0.9, linestyle='-')


class Live3DVisualizer:
    """Live 3D visualization of lights, camera, and user position."""
    
    def __init__(self, light_positions: Dict[str, np.ndarray], light_names: Dict[str, str],
                 camera_position: np.ndarray, camera_orientation: np.ndarray,
                 angle_threshold: float = POINTING_ANGLE_THRESHOLD):
        self.light_positions = light_positions
        self.light_names = light_names
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        self.angle_threshold = angle_threshold
        
        self.fig = None
        self.ax = None
        self.running = False
        self.update_lock = threading.Lock()
        self.current_user_pos = None
        self.current_pointing_vector = None
        self.active_lights = set()
        
        self.start()
    
    def start(self):
        """Start the visualization window."""
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.running = True
        self._update_plot()
    
    def _update_plot(self):
        """Update the 3D plot."""
        self.ax.clear()
        
        max_distance = 0
        if self.current_user_pos is not None:
            for pos in self.light_positions.values():
                dist = np.linalg.norm(pos - self.current_user_pos)
                max_distance = max(max_distance, dist)
        
        cone_length = max(2.0, max_distance * 1.2) if max_distance > 0 else 3.0
        
        for light_id, pos in self.light_positions.items():
            name = self.light_names.get(light_id, f"Light {light_id}")
            is_active = light_id in self.active_lights
            
            color = 'red' if is_active else 'blue'
            size = 350 if is_active else 200
            edge_color = 'darkred' if is_active else 'darkblue'
            self.ax.scatter(pos[0], pos[1], pos[2], s=size, c=color, alpha=0.8, 
                          edgecolors=edge_color, linewidth=2)
            self.ax.text(pos[0], pos[1], pos[2], f"  {light_id}\n  {name[:15]}", 
                        fontsize=7, color='white' if is_active else 'black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))
        
        self.ax.scatter(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                       s=400, c='red', marker='^', label='Camera', edgecolors='black', linewidth=2)
        
        arrow_length = 0.3
        self.ax.quiver(self.camera_position[0], self.camera_position[1], self.camera_position[2],
                      self.camera_orientation[0] * arrow_length,
                      self.camera_orientation[1] * arrow_length,
                      self.camera_orientation[2] * arrow_length,
                      color='red', arrow_length_ratio=0.3, linewidth=2)
        
        if self.current_user_pos is not None:
            self.ax.scatter(self.current_user_pos[0], self.current_user_pos[1], self.current_user_pos[2],
                           s=600, c='orange', marker='o', label='You', edgecolors='black', linewidth=3)
            
            if self.current_pointing_vector is not None:
                draw_cone_3d(self.ax, 
                            self.current_user_pos, 
                            self.current_pointing_vector,
                            self.angle_threshold,
                            cone_length,
                            color='yellow',
                            alpha=0.25,
                            resolution=30)
                
                axis_end = self.current_user_pos + self.current_pointing_vector * cone_length
                self.ax.plot([self.current_user_pos[0], axis_end[0]],
                            [self.current_user_pos[1], axis_end[1]],
                            [self.current_user_pos[2], axis_end[2]],
                            color='yellow', linewidth=3, alpha=0.8, linestyle='--')
        
        self.ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('Z Position', fontsize=12, fontweight='bold')
        self.ax.set_title('Live 3D Map - Orchestrator Control', fontsize=14, fontweight='bold')
        
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
    """Find lights within the pointing cone."""
    active_lights = set()
    threshold_cos = math.cos(math.radians(angle_threshold))
    
    for light_id, light_pos in light_positions.items():
        to_light = light_pos - user_pos
        distance = np.linalg.norm(to_light)
        
        if distance < 1e-6:
            continue
        
        to_light_normalized = to_light / distance
        cos_angle = np.dot(pointing_vector, to_light_normalized)
        
        if cos_angle >= threshold_cos:
            active_lights.add(light_id)
    
    return active_lights


def start_orchestrator_control(bridge: Bridge, light_positions: Dict[str, np.ndarray],
                               light_names: Dict[str, str], user_position: np.ndarray,
                               calib_manager: CalibrationManager, coord_mapper: CoordinateMapper):
    """Start the orchestrator light control system."""
    # Initialize trackers
    hand_tracker = HandTracker()
    pose_tracker = PoseTracker()
    
    # Initialize orchestrator light controller
    orchestrator_controller = OrchestratorLightController(bridge, list(light_positions.keys()))
    
    # Initialize live 3D visualizer
    visualizer = Live3DVisualizer(light_positions, light_names,
                                  calib_manager.camera_position, calib_manager.camera_orientation,
                                  angle_threshold=POINTING_ANGLE_THRESHOLD)
    
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
    
    visualization_update_interval = 0.1
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
            
            # Use manually entered position
            user_pos = user_position.copy() if user_position is not None else None
            pointing_vector = None
            active_lights = set()
            
            # Draw pose landmarks for visualization
            if pose_results.pose_landmarks:
                pose_tracker.draw_landmarks(frame, pose_results.pose_landmarks)
            
            # Get pointing vector from hand
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                camera_pointing = hand_tracker.get_pointing_vector(hand_landmarks, frame.shape)
                
                if camera_pointing is not None and user_pos is not None:
                    # Transform pointing vector to room coordinates
                    pointing_vector = coord_mapper.transform_to_room_coords(camera_pointing, scale=1.0)
                    pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)
                    
                    # Find lights in pointing cone
                    active_lights = find_lights_in_cone(user_pos, pointing_vector, light_positions)
                    
                    # Update lights
                    orchestrator_controller.update_lights(active_lights)
                
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
                active_names = [light_names.get(lid, f"Light {lid}") for lid in active_lights]
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
                dummy_positions = {lid: np.array([0.0, 0.0, 0.0]) for lid in light_positions.keys()}
                if calib_manager.calibrate_interactive(dummy_positions, light_names):
                    coord_mapper = CoordinateMapper(calib_manager.camera_position, calib_manager.camera_orientation)
                    visualizer = Live3DVisualizer(light_positions, light_names,
                                                  calib_manager.camera_position, calib_manager.camera_orientation,
                                                  angle_threshold=POINTING_ANGLE_THRESHOLD)
            elif key == ord('p'):
                # Update user position
                print("\nUpdating user position...")
                dummy_positions = {lid: np.array([0.0, 0.0, 0.0]) for lid in light_positions.keys()}
                new_pos = get_user_position_manual(dummy_positions, light_names,
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
        for light_id in orchestrator_controller.light_ids:
            orchestrator_controller.restore_light_state(light_id)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        visualizer.stop()
        print("\nOrchestrator control stopped.")


def get_user_position_manual(light_positions: Dict[str, np.ndarray], 
                             light_names: Dict[str, str],
                             camera_position: np.ndarray) -> Optional[np.ndarray]:
    """Get user position manually with 3D plot reference."""
    print("\n" + "="*60)
    print("USER POSITION SETUP")
    print("="*60)
    print("Enter your position in the room (X, Y, Z coordinates).")
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


def load_room_data(filename: str) -> Optional[Dict]:
    """Load room data from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None


def calibrate_light_by_pointing(light_id: str, light_name: str, light_controller: LightController,
                                hand_tracker: HandTracker, coord_mapper: CoordinateMapper,
                                user_position: np.ndarray, cap, calibration_duration: float = CALIBRATION_DURATION):
    """
    Calibrate a single light by tracking pointing direction.
    
    Returns:
        Average pointing direction in room coordinates, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Calibrating: {light_name} (ID: {light_id})")
    print(f"{'='*60}")
    print(f"Point at the RED light for {calibration_duration} seconds...")
    
    # Turn light red
    light_controller.set_light_red(light_id)
    time.sleep(0.5)  # Give light time to change
    
    # Collect pointing vectors
    pointing_vectors = []
    start_time = time.time()
    frame_count = 0
    
    print("\nStarting calibration...")
    
    while time.time() - start_time < calibration_duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand landmarks
        hand_results = hand_tracker.detect_landmarks(frame)
        
        if hand_results.multi_hand_landmarks:
            # Use first detected hand
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            camera_pointing = hand_tracker.get_pointing_vector(hand_landmarks, frame.shape)
            
            if camera_pointing is not None:
                # Transform pointing vector to room coordinates
                pointing_vector = coord_mapper.transform_to_room_coords(camera_pointing, scale=1.0)
                pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)  # Normalize
                pointing_vectors.append(pointing_vector)
            
            hand_tracker.draw_landmarks(frame, hand_landmarks)
        
        # Draw countdown
        elapsed = time.time() - start_time
        remaining = max(0, calibration_duration - elapsed)
        cv2.putText(frame, f"Calibrating: {light_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Time remaining: {remaining:.1f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Samples collected: {len(pointing_vectors)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Point at the RED light!", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Light Calibration', frame)
        cv2.waitKey(1)
        frame_count += 1
    
    # Restore light
    light_controller.restore_light_state(light_id)
    
    if len(pointing_vectors) == 0:
        print(f"⚠ No pointing data collected for {light_name}")
        return None
    
    # Calculate average pointing direction
    pointing_array = np.array(pointing_vectors)
    avg_pointing = np.mean(pointing_array, axis=0)
    avg_pointing = avg_pointing / np.linalg.norm(avg_pointing)
    
    print(f"✓ Collected {len(pointing_vectors)} pointing samples")
    print(f"✓ Average pointing direction: [{avg_pointing[0]:.3f}, {avg_pointing[1]:.3f}, {avg_pointing[2]:.3f}]")
    
    return avg_pointing


def calculate_light_position(user_position: np.ndarray, pointing_direction: np.ndarray,
                            estimated_distance: float) -> np.ndarray:
    """
    Calculate light position from user position and pointing direction.
    
    Args:
        user_position: User's 3D position
        pointing_direction: Normalized pointing direction vector
        estimated_distance: Estimated distance to light
    
    Returns:
        Calculated light position
    """
    return user_position + pointing_direction * estimated_distance


def estimate_light_distance(light_id: str, light_name: str) -> float:
    """Ask user to estimate distance to a light."""
    while True:
        try:
            dist_input = input(f"\nEstimated distance to '{light_name}' (in meters, or press Enter for default 2.5m): ").strip()
            if not dist_input:
                return 2.5  # Default distance
            distance = float(dist_input)
            if distance > 0:
                return distance
            else:
                print("Distance must be positive.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            return 2.5  # Default on cancel


def save_calibrated_positions(calibrated_positions: Dict[str, np.ndarray], 
                              light_names: Dict[str, str],
                              user_position: Optional[np.ndarray] = None,
                              filename: str = LIGHT_POSITIONS_FILE):
    """Save calibrated light positions and optionally user position to file."""
    data = {
        'calibrated_positions': {lid: pos.tolist() for lid, pos in calibrated_positions.items()},
        'light_names': light_names,
        'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Also save user position if provided
    if user_position is not None:
        data['user_position'] = user_position.tolist()
        # Also save to separate user position file
        with open(USER_POSITION_FILE, 'w') as f:
            json.dump({'user_position': user_position.tolist()}, f, indent=2)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Calibrated positions saved to {filename}")


def load_saved_data(light_ids: List[str]) -> tuple:
    """
    Load saved user position and calibrated light positions.
    
    Returns:
        tuple: (user_position, calibrated_positions, light_names)
        user_position: np.ndarray or None
        calibrated_positions: Dict[str, np.ndarray] or None
        light_names: Dict[str, str] or None
    """
    user_position = None
    calibrated_positions = None
    light_names = None
    
    # Try to load from light positions file (which may contain user position)
    try:
        with open(LIGHT_POSITIONS_FILE, 'r') as f:
            data = json.load(f)
            if 'user_position' in data:
                user_position = np.array(data['user_position'])
            if 'calibrated_positions' in data:
                calibrated_positions = {
                    lid: np.array(pos) 
                    for lid, pos in data['calibrated_positions'].items()
                }
            if 'light_names' in data:
                light_names = data['light_names']
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: Could not load from {LIGHT_POSITIONS_FILE}: {e}")
    
    # Also try separate user position file
    if user_position is None:
        try:
            with open(USER_POSITION_FILE, 'r') as f:
                data = json.load(f)
                if 'user_position' in data:
                    user_position = np.array(data['user_position'])
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Could not load from {USER_POSITION_FILE}: {e}")
    
    return user_position, calibrated_positions, light_names


def visualize_calibration(calibrated_positions: Dict[str, np.ndarray],
                         light_names: Dict[str, str],
                         user_position: np.ndarray,
                         camera_position: np.ndarray):
    """Visualize the calibrated light positions (non-blocking)."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot lights
    for light_id, pos in calibrated_positions.items():
        name = light_names.get(light_id, f"Light {light_id}")
        ax.scatter(pos[0], pos[1], pos[2], s=300, alpha=0.8, c='blue', edgecolors='black', linewidth=2)
        ax.text(pos[0], pos[1], pos[2], f"  {light_id}\n  {name[:20]}", fontsize=8)
    
    # Plot user position
    ax.scatter(user_position[0], user_position[1], user_position[2],
              s=500, c='orange', marker='o', label='You', edgecolors='black', linewidth=3)
    
    # Plot camera position
    ax.scatter(camera_position[0], camera_position[1], camera_position[2],
              s=400, c='red', marker='^', label='Camera', edgecolors='black', linewidth=2)
    
    # Set equal aspect ratio
    all_positions = list(calibrated_positions.values())
    all_positions.append(user_position)
    all_positions.append(camera_position)
    if all_positions:
        positions_array = np.array(all_positions)
        max_range = np.array([
            positions_array[:, 0].max() - positions_array[:, 0].min(),
            positions_array[:, 1].max() - positions_array[:, 1].min(),
            positions_array[:, 2].max() - positions_array[:, 2].min()
        ]).max() / 2.0
        
        mid = positions_array.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position', fontsize=12, fontweight='bold')
    ax.set_title('Calibrated Light Positions', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Show plot non-blocking
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure window is displayed
    
    return fig


def main():
    print("=" * 60)
    print("Light Calibration by Pointing")
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
    
    # Extract light information
    light_ids = [str(light['id']) for light in room_data.get('lights', [])]
    light_names = {str(light['id']): light['name'] for light in room_data.get('lights', [])}
    
    if not light_ids:
        print("\n⚠ No lights found in room data.")
        return
    
    print(f"\n✓ Found {len(light_ids)} lights")
    
    # Load saved data
    print("\nLoading saved calibration data...")
    saved_user_pos, saved_calibrated_positions, saved_light_names = load_saved_data(light_ids)
    
    # Check if we have all required data to skip calibration
    has_all_data = (saved_user_pos is not None and 
                   saved_calibrated_positions and 
                   len(saved_calibrated_positions) == len(light_ids))
    
    if has_all_data:
        print(f"✓ Found saved user position: [{saved_user_pos[0]:.2f}, {saved_user_pos[1]:.2f}, {saved_user_pos[2]:.2f}]")
        print(f"✓ Found {len(saved_calibrated_positions)} saved light positions")
        print("\n" + "=" * 60)
        skip_calibration = input("Skip calibration and use previous settings? (y/n): ").strip().lower()
        
        if skip_calibration == 'y':
            # Skip to orchestrator control
            print("\n" + "=" * 60)
            print("Skipping calibration - Using previous settings")
            print("=" * 60)
            
            # Load camera calibration
            calib_manager = CalibrationManager()
            if not calib_manager.load_calibration():
                print("\n⚠ No camera calibration found. Please run calibration first.")
                return
            
            print(f"✓ Camera position: {calib_manager.camera_position}")
            print(f"✓ Camera orientation: {calib_manager.camera_orientation}")
            print(f"✓ User position: [{saved_user_pos[0]:.2f}, {saved_user_pos[1]:.2f}, {saved_user_pos[2]:.2f}]")
            print(f"✓ Calibrated lights: {len(saved_calibrated_positions)}")
            
            # Initialize coordinate mapper
            coord_mapper = CoordinateMapper(calib_manager.camera_position, calib_manager.camera_orientation)
            
            # Start orchestrator control directly
            print("\n" + "=" * 60)
            print("Starting Orchestrator Light Control...")
            print("=" * 60)
            start_orchestrator_control(bridge, saved_calibrated_positions, light_names, 
                                      saved_user_pos, calib_manager, coord_mapper)
            return
    elif saved_user_pos is not None or saved_calibrated_positions:
        if saved_user_pos is not None:
            print(f"✓ Found saved user position: [{saved_user_pos[0]:.2f}, {saved_user_pos[1]:.2f}, {saved_user_pos[2]:.2f}]")
        if saved_calibrated_positions:
            print(f"✓ Found {len(saved_calibrated_positions)} saved light positions")
    
    # Calibration setup
    calib_manager = CalibrationManager()
    if not calib_manager.load_calibration():
        print("\nNo calibration found. Starting camera calibration...")
        # Create dummy positions for calibration reference
        dummy_positions = {lid: np.array([0.0, 0.0, 0.0]) for lid in light_ids}
        if not calib_manager.calibrate_interactive(dummy_positions, light_names):
            print("Calibration cancelled.")
            return
    else:
        print("\n✓ Loaded existing camera calibration")
        print(f"  Camera position: {calib_manager.camera_position}")
        print(f"  Camera orientation: {calib_manager.camera_orientation}")
        
        recalibrate = input("\nRecalibrate camera? (y/n): ").strip().lower()
        if recalibrate == 'y':
            dummy_positions = {lid: np.array([0.0, 0.0, 0.0]) for lid in light_ids}
            if not calib_manager.calibrate_interactive(dummy_positions, light_names):
                print("Calibration cancelled.")
                return
    
    # Get user position
    if saved_user_pos is not None:
        use_saved = input(f"\nUse saved user position [{saved_user_pos[0]:.2f}, {saved_user_pos[1]:.2f}, {saved_user_pos[2]:.2f}]? (y/n): ").strip().lower()
        if use_saved == 'y':
            user_position = saved_user_pos
            print("✓ Using saved user position")
        else:
            dummy_positions = {lid: np.array([0.0, 0.0, 0.0]) for lid in light_ids}
            user_position = get_user_position_manual(dummy_positions, light_names, calib_manager.camera_position)
            if user_position is None:
                print("User position setup cancelled.")
                return
    else:
        dummy_positions = {lid: np.array([0.0, 0.0, 0.0]) for lid in light_ids}
        user_position = get_user_position_manual(dummy_positions, light_names, calib_manager.camera_position)
        if user_position is None:
            print("User position setup cancelled.")
            return
    
    # Initialize coordinate mapper
    coord_mapper = CoordinateMapper(calib_manager.camera_position, calib_manager.camera_orientation)
    
    # Initialize light controller
    light_controller = LightController(bridge, light_ids)
    
    # Start with saved positions if available
    calibrated_positions = saved_calibrated_positions.copy() if saved_calibrated_positions else {}
    
    # Determine which lights need calibration
    lights_to_calibrate = []
    for light_id in light_ids:
        light_name = light_names[light_id]
        if light_id in calibrated_positions:
            saved_pos = calibrated_positions[light_id]
            recalibrate = input(f"\nLight '{light_name}' already calibrated at [{saved_pos[0]:.2f}, {saved_pos[1]:.2f}, {saved_pos[2]:.2f}]. Recalibrate? (y/n): ").strip().lower()
            if recalibrate == 'y':
                lights_to_calibrate.append(light_id)
        else:
            lights_to_calibrate.append(light_id)
    
    if not lights_to_calibrate:
        print("\n✓ All lights are already calibrated. No calibration needed.")
        cap = None  # No camera needed
    else:
        # Initialize hand tracker and camera only if we need to calibrate
        hand_tracker = HandTracker()
        
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
        print("CALIBRATION PROCESS")
        print("=" * 60)
        print(f"Calibrating {len(lights_to_calibrate)} light(s):")
        for light_id in lights_to_calibrate:
            print(f"  - {light_names[light_id]}")
        print("\nFor each light:")
        print("  1. The light will turn RED")
        print("  2. Point at it with your hand for 5 seconds")
        print("  3. You'll be asked for the estimated distance")
        print("=" * 60)
        
        input("\nPress Enter to start calibration...")
        
        # Countdown before starting
        print("\nStarting in 3 seconds... Get ready!")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("  Go!\n")
        
        # Calibrate each light that needs it
        pointing_directions = {}
        
        for i, light_id in enumerate(lights_to_calibrate, 1):
            light_name = light_names[light_id]
            print(f"\n[{i}/{len(lights_to_calibrate)}] Calibrating light: {light_name}")
            
            # Calibrate by pointing
            pointing_dir = calibrate_light_by_pointing(
                light_id, light_name, light_controller, hand_tracker,
                coord_mapper, user_position, cap, CALIBRATION_DURATION
            )
            
            if pointing_dir is None:
                print(f"⚠ Skipping {light_name} - no pointing data collected")
                # Still wait for Enter to continue
                if i < len(lights_to_calibrate):
                    input("\nPress Enter to continue to next light...")
                continue
            
            pointing_directions[light_id] = pointing_dir
            
            # Get distance estimate
            distance = estimate_light_distance(light_id, light_name)
            
            # Calculate light position
            light_pos = calculate_light_position(user_position, pointing_dir, distance)
            calibrated_positions[light_id] = light_pos
            
            print(f"✓ Calculated position for {light_name}: [{light_pos[0]:.2f}, {light_pos[1]:.2f}, {light_pos[2]:.2f}]")
            
            # Wait for Enter to continue to next light (except for the last one)
            if i < len(lights_to_calibrate):
                input("\nPress Enter to continue to next light...")
    
    # Cleanup camera (only if we opened it)
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    
    if not calibrated_positions:
        print("\n⚠ No lights are calibrated.")
        light_controller.restore_all()
        return
    
    # Save calibrated positions (including user position)
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Total calibrated lights: {len(calibrated_positions)}")
    if lights_to_calibrate:
        print(f"Newly calibrated: {len([lid for lid in lights_to_calibrate if lid in calibrated_positions])}")
    
    save_calibrated_positions(calibrated_positions, light_names, user_position)
    
    # Visualize results
    print("\nShowing calibration visualization...")
    print("(You can close the plot window or press Enter to continue)")
    fig = visualize_calibration(calibrated_positions, light_names, user_position, calib_manager.camera_position)
    
    # Wait for Enter to continue (works whether window is open or closed)
    try:
        input("\nPress Enter to start orchestrator control...")
    except KeyboardInterrupt:
        pass
    finally:
        # Close the plot window if still open
        plt.close(fig)
    
    # Start orchestrator control
    print("\n" + "=" * 60)
    print("Starting Orchestrator Light Control...")
    print("=" * 60)
    start_orchestrator_control(bridge, calibrated_positions, light_names, 
                               user_position, calib_manager, coord_mapper)
    
    # Restore all lights
    light_controller.restore_all()
    print("\nCalibration complete!")


if __name__ == "__main__":
    main()

