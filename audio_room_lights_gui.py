"""
Audio-reactive room lights with GUI visualization - brightness based on volume, color based on frequency.
Features multi-user recording with real-time plots showing amplitude vs frequency and time series.
"""

import numpy as np
import sounddevice as sd
import colorsys
import time
import threading
from phue import Bridge, Group
from pynput import keyboard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import queue

BRIDGE_IP = "192.168.1.2"

# Audio settings
SAMPLE_RATE = 44100
CHUNK_SIZE = 4096  # Number of samples per chunk
CHANNELS = 1  # Mono audio

# Frequency range for color mapping (Hz)
MIN_FREQ = 80   # Low frequency = blue
MAX_FREQ = 2000  # High frequency = red

# Volume range for brightness mapping
MIN_VOLUME_DB = -60  # Quiet = dim
MAX_VOLUME_DB = -10   # Loud = bright

# User colors for plotting (distinct colors for up to 10 users)
USER_COLORS = [
    '#FF0000',  # Red - User 1
    '#00FF00',  # Green - User 2
    '#0000FF',  # Blue - User 3
    '#FFFF00',  # Yellow - User 4
    '#FF00FF',  # Magenta - User 5
    '#00FFFF',  # Cyan - User 6
    '#FFA500',  # Orange - User 7
    '#800080',  # Purple - User 8
    '#FFC0CB',  # Pink - User 9
    '#A52A2A',  # Brown - User 10
]


def frequency_to_hue(frequency):
    """
    Convert frequency to hue color (blue to red).
    
    Args:
        frequency: Frequency in Hz
    
    Returns:
        hue: Hue value (0-65535) for Hue lights
    """
    # Clamp frequency to range
    freq = max(MIN_FREQ, min(MAX_FREQ, frequency))
    
    # Normalize to 0-1
    normalized = (freq - MIN_FREQ) / (MAX_FREQ - MIN_FREQ)
    
    # Map to hue: blue (240°) to red (0°/360°)
    # In HSV: blue is 240/360 = 0.667, red is 0/360 = 0.0
    # We want to go from blue (high) to red (low), so invert
    hue_normalized = 1.0 - normalized  # Invert so low freq = blue, high freq = red
    
    # Convert to 0-1 range for HSV (blue=0.667, red=0.0)
    # We need to map 0-1 to the range 0.667 (blue) to 0.0 (red)
    # But we want it to wrap around: blue -> cyan -> green -> yellow -> red
    # So: 0.0 -> 0.667 (blue), 1.0 -> 0.0 (red)
    hue_hsv = (0.667 - (hue_normalized * 0.667)) % 1.0
    
    # Convert to Hue color space (0-65535)
    hue = int(hue_hsv * 65535)
    
    return hue


def hue_to_color_name(hue_value):
    """
    Convert hue value to color name and percentage.
    
    Args:
        hue_value: Hue value (0-65535)
    
    Returns:
        tuple: (color_name, percentage) where percentage is 0-100
    """
    # Convert to HSV normalized (0-1)
    hue_normalized = hue_value / 65535.0
    
    # Map to color names with percentage within each color range
    # Red: 0.0-0.05 or 0.95-1.0
    if hue_normalized < 0.05:
        return "Red", int((hue_normalized / 0.05) * 100)
    elif hue_normalized >= 0.95:
        return "Red", int(((hue_normalized - 0.95) / 0.05) * 100)
    # Orange: 0.05-0.15
    elif hue_normalized < 0.15:
        return "Orange", int(((hue_normalized - 0.05) / 0.10) * 100)
    # Yellow: 0.15-0.25
    elif hue_normalized < 0.25:
        return "Yellow", int(((hue_normalized - 0.15) / 0.10) * 100)
    # Green: 0.25-0.40
    elif hue_normalized < 0.40:
        return "Green", int(((hue_normalized - 0.25) / 0.15) * 100)
    # Cyan: 0.40-0.55
    elif hue_normalized < 0.55:
        return "Cyan", int(((hue_normalized - 0.40) / 0.15) * 100)
    # Blue: 0.55-0.70
    elif hue_normalized < 0.70:
        return "Blue", int(((hue_normalized - 0.55) / 0.15) * 100)
    # Indigo: 0.70-0.80
    elif hue_normalized < 0.80:
        return "Indigo", int(((hue_normalized - 0.70) / 0.10) * 100)
    # Purple: 0.80-0.95
    else:
        return "Purple", int(((hue_normalized - 0.80) / 0.15) * 100)


def volume_to_brightness(volume_db):
    """
    Convert volume (dB) to brightness.
    
    Args:
        volume_db: Volume in decibels
    
    Returns:
        brightness: Brightness value (1-254) for Hue lights
    """
    # Clamp volume to range
    vol = max(MIN_VOLUME_DB, min(MAX_VOLUME_DB, volume_db))
    
    # Normalize to 0-1
    normalized = (vol - MIN_VOLUME_DB) / (MAX_VOLUME_DB - MIN_VOLUME_DB)
    
    # Map to brightness range (1-254)
    brightness = int(normalized * 253) + 1
    
    return brightness


def get_dominant_frequency(audio_data, sample_rate):
    """
    Get the dominant frequency from audio data using FFT.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
    
    Returns:
        frequency: Dominant frequency in Hz
    """
    # Apply window function to reduce spectral leakage
    windowed = audio_data * np.hanning(len(audio_data))
    
    # Perform FFT
    fft = np.fft.rfft(windowed)
    fft_magnitude = np.abs(fft)
    
    # Get frequency bins
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    
    # Find dominant frequency (peak in magnitude)
    # Only consider frequencies in our range
    mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
    if not np.any(mask):
        return (MIN_FREQ + MAX_FREQ) / 2  # Default to middle
    
    masked_freqs = freqs[mask]
    masked_magnitude = fft_magnitude[mask]
    
    # Find peak
    peak_idx = np.argmax(masked_magnitude)
    dominant_freq = masked_freqs[peak_idx]
    
    return dominant_freq


def get_volume_db(audio_data):
    """
    Calculate RMS volume in decibels.
    
    Args:
        audio_data: Audio samples
    
    Returns:
        volume_db: Volume in decibels
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(audio_data**2))
    
    # Convert to dB (avoid log(0))
    if rms < 1e-10:
        return MIN_VOLUME_DB
    
    # Reference level for dB calculation
    volume_db = 20 * np.log10(rms)
    
    return volume_db


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


class UserRecording:
    """Stores recording data for a single user."""
    
    def __init__(self, user_id: int, color: str):
        self.user_id = user_id
        self.color = color
        self.recordings = []  # List of completed recordings, each is a list of (frequency, amplitude) tuples
        self.current_recording = []  # Current active recording
        self.is_recording = False
        self.recording_start_time = None
        self.lock = threading.Lock()
    
    def start_recording(self):
        """Start a new recording."""
        with self.lock:
            self.is_recording = True
            self.current_recording = []
            self.recording_start_time = time.time()
    
    def add_sample(self, frequency: float, amplitude: float):
        """Add a sample to the current recording."""
        with self.lock:
            if self.is_recording:
                elapsed = time.time() - self.recording_start_time
                self.current_recording.append((elapsed, frequency, amplitude))
    
    def stop_recording(self):
        """Stop recording and save it."""
        with self.lock:
            if self.is_recording and len(self.current_recording) > 0:
                # Save the completed recording
                self.recordings.append(self.current_recording.copy())
                self.current_recording = []
            self.is_recording = False
            self.recording_start_time = None
    
    def get_current_recording_data(self):
        """Get current recording data (thread-safe)."""
        with self.lock:
            if self.is_recording and len(self.current_recording) > 0:
                times = [s[0] for s in self.current_recording]
                frequencies = [s[1] for s in self.current_recording]
                amplitudes = [s[2] for s in self.current_recording]
                return times, frequencies, amplitudes
            return [], [], []
    
    def get_completed_recordings_stats(self):
        """Get statistics for all completed recordings."""
        with self.lock:
            stats = []
            for recording in self.recordings:
                if len(recording) > 0:
                    frequencies = [s[1] for s in recording]
                    amplitudes = [s[2] for s in recording]
                    avg_freq = np.mean(frequencies)
                    std_freq = np.std(frequencies)
                    avg_amp = np.mean(amplitudes)
                    std_amp = np.std(amplitudes)
                    stats.append({
                        'avg_freq': avg_freq,
                        'std_freq': std_freq,
                        'avg_amp': avg_amp,
                        'std_amp': std_amp,
                        'color': self.color
                    })
            return stats


class AudioRoomControllerGUI:
    """Controller for room lights with GUI visualization."""
    
    def __init__(self, bridge: Bridge, group_id: str):
        """
        Initialize audio-reactive room controller with GUI.
        
        Args:
            bridge: Connected Hue Bridge
            group_id: Group/room ID to control
        """
        self.bridge = bridge
        self.group_id = group_id
        self.group = Group(bridge, group_id)
        self.last_hue = None
        self.last_brightness = None
        self.update_lock = threading.Lock()
        self.running = True
        
        # User recordings (up to 10 users)
        self.users = {}
        for i in range(1, 11):
            self.users[i] = UserRecording(i, USER_COLORS[i - 1])
        
        self.active_user = None  # Currently recording user
        
        # Audio averaging for lights
        self.audio_samples = []  # Store (frequency, volume_db) tuples
        self.sample_start_time = None
        self.avg_window_seconds = 2.0  # Average over 2 seconds
        
        # Data queue for GUI updates
        self.data_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup matplotlib GUI with two subplots."""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.fig.suptitle('Audio-Reactive Room Lights - Multi-User Recording', fontsize=14, fontweight='bold')
        
        # Plot 1: Amplitude vs Frequency (with averages and std dev)
        self.ax1.set_xlabel('Amplitude (dB)', fontsize=11)
        self.ax1.set_ylabel('Frequency (Hz)', fontsize=11)
        self.ax1.set_title('Amplitude vs Frequency (All Users)', fontsize=12, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(MIN_VOLUME_DB, MAX_VOLUME_DB)
        self.ax1.set_ylim(MIN_FREQ, MAX_FREQ)
        
        # Plot 2: Time series of frequency
        self.ax2.set_xlabel('Time (s)', fontsize=11)
        self.ax2.set_ylabel('Frequency (Hz)', fontsize=11)
        self.ax2.set_title('Frequency Time Series (Current Recording)', fontsize=12, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(MIN_FREQ, MAX_FREQ)
        
        # Store plot artists
        self.plot1_artists = {}  # User ID -> list of artists
        self.plot2_line = None
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_gui(self):
        """Update GUI plots with latest data."""
        try:
            # Update plot 1: Amplitude vs Frequency with averages and std dev
            self.ax1.clear()
            self.ax1.set_xlabel('Amplitude (dB)', fontsize=11)
            self.ax1.set_ylabel('Frequency (Hz)', fontsize=11)
            self.ax1.set_title('Amplitude vs Frequency (All Users)', fontsize=12, fontweight='bold')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_xlim(MIN_VOLUME_DB, MAX_VOLUME_DB)
            self.ax1.set_ylim(MIN_FREQ, MAX_FREQ)
            
            # Plot completed recordings for all users
            for user_id, user_recording in self.users.items():
                stats = user_recording.get_completed_recordings_stats()
                for i, stat in enumerate(stats):
                    # Plot average point (label only first recording per user)
                    label = f'User {user_id}' if i == 0 else ''
                    self.ax1.scatter(stat['avg_amp'], stat['avg_freq'], 
                                   color=stat['color'], s=100, alpha=0.7,
                                   label=label)
                    
                    # Plot error bars (std dev)
                    self.ax1.errorbar(stat['avg_amp'], stat['avg_freq'],
                                    xerr=stat['std_amp'], yerr=stat['std_freq'],
                                    color=stat['color'], alpha=0.5, capsize=5, capthick=2)
            
            # Add legend if there are any recordings
            handles, labels = self.ax1.get_legend_handles_labels()
            if handles:
                # Remove duplicate labels
                by_label = dict(zip(labels, handles))
                self.ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
            
            # Update plot 2: Time series for current recording
            self.ax2.clear()
            self.ax2.set_xlabel('Time (s)', fontsize=11)
            self.ax2.set_ylabel('Frequency (Hz)', fontsize=11)
            self.ax2.set_title('Frequency Time Series (Current Recording)', fontsize=12, fontweight='bold')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(MIN_FREQ, MAX_FREQ)
            
            if self.active_user and self.active_user in self.users:
                times, frequencies, _ = self.users[self.active_user].get_current_recording_data()
                if len(times) > 0:
                    self.ax2.plot(times, frequencies, 
                                color=self.users[self.active_user].color,
                                linewidth=2, label=f'User {self.active_user}')
                    self.ax2.legend(loc='upper right', fontsize=9)
                    if len(times) > 0:
                        self.ax2.set_xlim(0, max(times) + 0.5)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating GUI: {e}")
    
    def _update_lights(self, hue: int, brightness: int):
        """Update lights in a separate thread."""
        try:
            # Turn on the group first
            if not self.group.on:
                self.group.on = True
            
            # Set color and brightness
            self.group.hue = hue
            self.group.saturation = 254  # Full saturation for vibrant colors
            self.group.brightness = brightness
        except Exception as e:
            print(f"Error updating lights: {e}")
    
    def add_audio_sample(self, frequency: float, volume_db: float):
        """
        Add an audio sample for averaging and recording.
        
        Args:
            frequency: Dominant frequency in Hz
            volume_db: Volume in decibels
        """
        current_time = time.time()
        
        # Add to current user recording if active
        if self.active_user and self.active_user in self.users:
            self.users[self.active_user].add_sample(frequency, volume_db)
        
        # Handle light updates (2-second averaging)
        if self.sample_start_time is None:
            self.sample_start_time = current_time
        
        self.audio_samples.append((frequency, volume_db))
        
        # Check if we've collected enough samples (2 seconds worth)
        elapsed = current_time - self.sample_start_time
        if elapsed >= self.avg_window_seconds:
            # Calculate averages
            avg_frequency = np.mean([s[0] for s in self.audio_samples])
            avg_volume_db = np.mean([s[1] for s in self.audio_samples])
            
            # Convert to hue and brightness
            hue = frequency_to_hue(avg_frequency)
            brightness = volume_to_brightness(avg_volume_db)
            
            # Reset for next window
            self.audio_samples = []
            self.sample_start_time = None
            
            # Update lights
            with self.update_lock:
                self.last_hue = hue
                self.last_brightness = brightness
            
            # Update lights in background thread
            thread = threading.Thread(target=self._update_lights, args=(hue, brightness))
            thread.daemon = True
            thread.start()
    
    def start_user_recording(self, user_id: int):
        """Start recording for a specific user."""
        if user_id in self.users:
            # Stop previous recording if any
            if self.active_user and self.active_user in self.users:
                self.users[self.active_user].stop_recording()
            
            # Start new recording
            self.users[user_id].start_recording()
            self.active_user = user_id
            print(f"🎤 User {user_id} started recording...")
    
    def stop_user_recording(self, user_id: int):
        """Stop recording for a specific user."""
        if user_id in self.users and user_id == self.active_user:
            self.users[user_id].stop_recording()
            stats = self.users[user_id].get_completed_recordings_stats()
            if stats:
                latest = stats[-1]
                print(f"✓ User {user_id} recording saved!")
                print(f"  Average Frequency: {latest['avg_freq']:.1f} ± {latest['std_freq']:.1f} Hz")
                print(f"  Average Amplitude: {latest['avg_amp']:.1f} ± {latest['std_amp']:.1f} dB")
            self.active_user = None


def audio_callback(indata, frames, time_info, status, controller):
    """Callback function for audio input."""
    if not controller.running:
        return
    
    if status:
        print(f"Audio status: {status}")
    
    # Get audio data
    audio_data = indata[:, 0]  # Mono channel
    
    # Analyze audio
    frequency = get_dominant_frequency(audio_data, SAMPLE_RATE)
    volume_db = get_volume_db(audio_data)
    
    # Add sample for averaging and recording
    controller.add_audio_sample(frequency, volume_db)


def list_audio_devices():
    """List available audio input devices."""
    print("\nAvailable audio input devices:")
    print("=" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  {i}. {device['name']}{default}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Audio-Reactive Room Lights with GUI")
    print("=" * 60)
    
    # List audio devices
    list_audio_devices()
    
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
    
    # Initialize controller
    print(f"\nInitializing audio control for '{selected_room['name']}'...")
    controller = AudioRoomControllerGUI(bridge, selected_room['id'])
    
    print("\n" + "=" * 60)
    print("Audio Control Active!")
    print("=" * 60)
    print("Controls:")
    print("  - Press 1-9,0 to start/stop recording for User 1-10")
    print("  - Volume controls brightness (louder = brighter)")
    print("  - Frequency controls color (low = blue, high = red)")
    print("  - Press ESC to stop")
    print("=" * 60)
    print("\nWaiting for user input...\n")
    
    # Track which keys are currently pressed (to avoid repeat triggers)
    pressed_keys = set()
    
    def on_press(key):
        """Handle key press events."""
        try:
            # Handle number keys (1-9, 0 for user 10)
            if hasattr(key, 'char') and key.char:
                if key.char in '1234567890':
                    user_id = int(key.char) if key.char != '0' else 10
                    # Only process if key wasn't already pressed (avoid repeat triggers)
                    if user_id not in pressed_keys:
                        pressed_keys.add(user_id)
                        user_recording = controller.users[user_id]
                        
                        if user_recording.is_recording:
                            # Stop recording if this user is currently recording
                            controller.stop_user_recording(user_id)
                        else:
                            # Start recording for this user
                            controller.start_user_recording(user_id)
        except AttributeError:
            pass
    
    def on_release(key):
        """Handle key release events."""
        try:
            if hasattr(key, 'char') and key.char:
                if key.char in '1234567890':
                    user_id = int(key.char) if key.char != '0' else 10
                    pressed_keys.discard(user_id)
        except AttributeError:
            pass
        
        if key == keyboard.Key.esc:
            controller.running = False
            return False
    
    # Start keyboard listener in background
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    try:
        # Start audio stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=CHUNK_SIZE,
            callback=lambda indata, frames, time_info, status: 
                audio_callback(indata, frames, time_info, status, controller),
            dtype=np.float32
        ):
            # GUI update loop
            last_gui_update = time.time()
            gui_update_interval = 0.1  # Update GUI every 100ms
            
            while controller.running:
                current_time = time.time()
                
                # Update GUI periodically
                if current_time - last_gui_update >= gui_update_interval:
                    controller.update_gui()
                    last_gui_update = current_time
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
    
    except KeyboardInterrupt:
        print("\n\nStopping audio stream...")
        controller.running = False
        listener.stop()
        time.sleep(0.5)
        print("Audio control stopped. Goodbye!")
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()

