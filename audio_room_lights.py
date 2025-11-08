"""
Audio-reactive room lights - brightness based on volume, color based on frequency.
"""

import numpy as np
import sounddevice as sd
import colorsys
import time
import threading
from phue import Bridge, Group
from pynput import keyboard

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


class AudioRoomController:
    """Controller for room lights based on audio."""
    
    def __init__(self, bridge: Bridge, group_id: str):
        """
        Initialize audio-reactive room controller.
        
        Args:
            bridge: Connected Hue Bridge
            group_id: Group/room ID to control
        """
        self.bridge = bridge
        self.group_id = group_id
        self.group = Group(bridge, group_id)
        self.last_hue = None
        self.last_brightness = None
        self.last_update_time = 0
        self.update_lock = threading.Lock()
        self.running = True
        
        # Audio averaging
        self.audio_samples = []  # Store (frequency, volume_db) tuples
        self.sample_start_time = None
        self.avg_window_seconds = 2.0  # Average over 2 seconds
        
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
        Add an audio sample for averaging.
        
        Args:
            frequency: Dominant frequency in Hz
            volume_db: Volume in decibels
        
        Returns:
            tuple: (hue, brightness, avg_frequency, avg_volume_db) if ready to update, None otherwise
        """
        current_time = time.time()
        
        # Initialize start time when first sample arrives
        if self.sample_start_time is None:
            self.sample_start_time = current_time
        
        # Add sample
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
                self.last_update_time = current_time
            
            # Update lights in background thread
            thread = threading.Thread(target=self._update_lights, args=(hue, brightness))
            thread.daemon = True
            thread.start()
            
            return hue, brightness, avg_frequency, avg_volume_db
        
        return None, None, None, None
    
    def reset_audio_collection(self):
        """
        Reset audio sample collection (called when SPACE is released).
        
        Returns:
            tuple: (hue, brightness, avg_frequency, avg_volume_db) if samples were processed, None otherwise
        """
        current_time = time.time()
        
        # If we have samples but haven't reached 2 seconds, process them anyway
        if self.audio_samples and self.sample_start_time is not None:
            elapsed = current_time - self.sample_start_time
            if elapsed > 0.1:  # Only process if we have at least 0.1 seconds of data
                # Calculate averages
                avg_frequency = np.mean([s[0] for s in self.audio_samples])
                avg_volume_db = np.mean([s[1] for s in self.audio_samples])
                
                # Convert to hue and brightness
                hue = frequency_to_hue(avg_frequency)
                brightness = volume_to_brightness(avg_volume_db)
                
                # Update lights
                with self.update_lock:
                    self.last_hue = hue
                    self.last_brightness = brightness
                    self.last_update_time = current_time
                
                # Update lights in background thread
                thread = threading.Thread(target=self._update_lights, args=(hue, brightness))
                thread.daemon = True
                thread.start()
                
                # Reset collection
                self.audio_samples = []
                self.sample_start_time = None
                
                return hue, brightness, avg_frequency, avg_volume_db
        
        # Reset collection
        self.audio_samples = []
        self.sample_start_time = None
        return None, None, None, None


def audio_callback(indata, frames, time_info, status, controller, space_pressed):
    """Callback function for audio input - collects samples when SPACE is pressed."""
    # Early return if controller is not running
    if not controller.running:
        return
    
    # CRITICAL: Only process audio if SPACE is pressed
    # Return immediately without any processing if space is not pressed
    if not space_pressed[0]:
        return  # No audio analysis, no light updates, nothing happens
    
    # Only reach here if SPACE is pressed
    if status:
        print(f"Audio status: {status}")
    
    # Get audio data
    audio_data = indata[:, 0]  # Mono channel
    
    # Analyze audio (only happens when SPACE is pressed)
    frequency = get_dominant_frequency(audio_data, SAMPLE_RATE)
    volume_db = get_volume_db(audio_data)
    
    # Add sample for averaging (will send update after 2 seconds)
    result = controller.add_audio_sample(frequency, volume_db)
    
    # Print status only if update happened (after 2 seconds)
    if result[0] is not None:
        hue, brightness, avg_freq, avg_vol = result
        color_name, color_percent = hue_to_color_name(hue)
        brightness_percent = int((brightness / 254.0) * 100)
        
        print(f"\n✓ Update sent!")
        print(f"  Averaged Frequency: {avg_freq:.1f} Hz")
        print(f"  Averaged Amplitude: {avg_vol:.1f} dB")
        print(f"  Color: {color_name} ({color_percent}%)")
        print(f"  Brightness: {brightness_percent}% ({brightness}/254)")
        print("  " + "-" * 50)


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
    print("Audio-Reactive Room Lights")
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
    controller = AudioRoomController(bridge, selected_room['id'])
    
    print("\n" + "=" * 60)
    print("Audio Control Active!")
    print("=" * 60)
    print("Controls:")
    print("  - Hold SPACE to activate audio control")
    print("  - Volume controls brightness (louder = brighter)")
    print("  - Frequency controls color (low = blue, high = red)")
    print("  - Press Ctrl+C to stop")
    print("=" * 60)
    print("\nWaiting for SPACE key to be pressed...\n")
    
    # Shared state for space key
    space_pressed = [False]
    
    def on_press(key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.space:
                space_pressed[0] = True
                print("🎤 Listening... (Collecting 2s average, release SPACE to stop)", end='\r')
        except AttributeError:
            pass
    
    def on_release(key):
        """Handle key release events."""
        try:
            if key == keyboard.Key.space:
                space_pressed[0] = False
                # Process any remaining samples when SPACE is released
                result = controller.reset_audio_collection()
                if result[0] is not None:
                    hue, brightness, avg_freq, avg_vol = result
                    color_name, color_percent = hue_to_color_name(hue)
                    brightness_percent = int((brightness / 254.0) * 100)
                    
                    print(f"\n✓ Final update sent (on release)!")
                    print(f"  Averaged Frequency: {avg_freq:.1f} Hz")
                    print(f"  Averaged Amplitude: {avg_vol:.1f} dB")
                    print(f"  Color: {color_name} ({color_percent}%)")
                    print(f"  Brightness: {brightness_percent}% ({brightness}/254)")
                    print("  " + "-" * 50)
                print("\n⏸️  Paused (Hold SPACE to listen)")
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
                audio_callback(indata, frames, time_info, status, controller, space_pressed),
            dtype=np.float32
        ):
            # Keep running until interrupted
            while controller.running:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping audio stream...")
        controller.running = False
        listener.stop()
        time.sleep(0.5)
        print("Audio control stopped. Goodbye!")


if __name__ == "__main__":
    main()

