"""
Disco mode script - makes lights in a room flash with vibrant colors like a disco!
"""

import colorsys
import random
import time
from phue import Bridge, Group

BRIDGE_IP = "192.168.1.2"
UPDATE_INTERVAL = 0.5  # Faster updates for disco effect (0.5 seconds)


def rgb_to_hue_hsv(r, g, b):
    """
    Convert RGB (0-255) to Hue color space (hue: 0-65535, sat: 0-254, bri: 0-254).
    """
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    
    hue = int(h * 65535)
    saturation = int(s * 254)
    brightness = int(v * 254)
    
    if brightness == 0 and (r > 0 or g > 0 or b > 0):
        brightness = 1
    
    return hue, saturation, brightness


def get_vibrant_color():
    """Generate a vibrant, saturated color (disco style)."""
    # Choose a random hue (color)
    hue = random.random()
    
    # High saturation for vibrant colors (0.8-1.0)
    saturation = random.uniform(0.8, 1.0)
    
    # Varying brightness for disco effect (0.6-1.0)
    brightness = random.uniform(0.6, 1.0)
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    
    # Convert to 0-255 range
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    
    return r, g, b


def get_classic_disco_color():
    """Generate classic disco colors (red, blue, green, yellow, magenta, cyan)."""
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 165, 0),    # Orange
        (255, 20, 147),   # Deep Pink
    ]
    return random.choice(colors)


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


def select_mode():
    """Let user select disco mode."""
    print("\nDisco Modes:")
    print("  1. Random Vibrant Colors (each light different)")
    print("  2. Classic Disco Colors (red, blue, green, etc.)")
    print("  3. Synchronized (all lights same color)")
    print("  4. Rainbow Wave (colors cycle through spectrum)")
    
    while True:
        try:
            choice = input("\nSelect mode (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def get_light_objects(bridge, light_ids):
    """Get Light objects for the given light IDs."""
    light_objects = []
    for light_id in light_ids:
        try:
            light = bridge[int(light_id)]
            if light.reachable:
                light_objects.append(light)
            else:
                print(f"  Warning: Light {light_id} ({light.name}) is not reachable")
        except Exception as e:
            print(f"  Warning: Could not access light {light_id}: {e}")
    return light_objects


def set_light_color(light, r, g, b):
    """Set a light to a specific RGB color."""
    try:
        hue, saturation, brightness = rgb_to_hue_hsv(r, g, b)
        light.on = True
        light.hue = hue
        light.saturation = saturation
        light.brightness = brightness
        return True
    except Exception as e:
        return False


def disco_mode_random(light_objects):
    """Disco mode: Each light gets a random vibrant color."""
    for light in light_objects:
        r, g, b = get_vibrant_color()
        set_light_color(light, r, g, b)
        print(f"  {light.name}: RGB({r}, {g}, {b})", end="  ")


def disco_mode_classic(light_objects):
    """Disco mode: Classic disco colors."""
    for light in light_objects:
        r, g, b = get_classic_disco_color()
        set_light_color(light, r, g, b)
        print(f"  {light.name}: RGB({r}, {g}, {b})", end="  ")


def disco_mode_sync(light_objects):
    """Disco mode: All lights synchronized to same color."""
    r, g, b = get_vibrant_color()
    for light in light_objects:
        set_light_color(light, r, g, b)
    print(f"  All lights: RGB({r}, {g}, {b})")


def disco_mode_rainbow(light_objects, iteration):
    """Disco mode: Rainbow wave effect."""
    num_lights = len(light_objects)
    if num_lights == 0:
        return
    
    for i, light in enumerate(light_objects):
        # Calculate hue offset for each light to create wave effect
        hue_offset = (iteration * 0.1 + i * (1.0 / num_lights)) % 1.0
        saturation = 1.0
        brightness = random.uniform(0.7, 1.0)
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue_offset, saturation, brightness)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        
        set_light_color(light, r, g, b)
    print(f"  Rainbow wave (iteration {iteration})")


def run_disco(bridge, room, light_objects, mode):
    """Run the disco effect."""
    mode_names = {
        1: "Random Vibrant Colors",
        2: "Classic Disco Colors",
        3: "Synchronized",
        4: "Rainbow Wave"
    }
    
    print(f"\n{'='*60}")
    print(f"🎉 DISCO MODE ACTIVATED! 🎉")
    print(f"{'='*60}")
    print(f"Room: {room['name']}")
    print(f"Mode: {mode_names[mode]}")
    print(f"Lights: {len(light_objects)}")
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print(f"{'='*60}")
    print("\nPress Ctrl+C to stop the disco!\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"Beat {iteration}: ", end="")
            
            if mode == 1:
                disco_mode_random(light_objects)
            elif mode == 2:
                disco_mode_classic(light_objects)
            elif mode == 3:
                disco_mode_sync(light_objects)
            elif mode == 4:
                disco_mode_rainbow(light_objects, iteration)
            
            print()  # New line
            time.sleep(UPDATE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n🎵 Disco stopped! 🎵")


def main():
    print("=" * 60)
    print("🎵 HUE DISCO MODE 🎵")
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
    
    # Get rooms
    print("\nFetching rooms...")
    rooms = get_rooms(bridge)
    
    if not rooms:
        print("No rooms found. Make sure you have rooms/groups set up in the Hue app.")
        return
    
    # Display and select room
    display_rooms(rooms)
    selected_room = select_room(rooms)
    
    if not selected_room:
        return
    
    # Select disco mode
    mode = select_mode()
    if mode is None:
        return
    
    # Get light objects
    print(f"\nLoading lights in room '{selected_room['name']}'...")
    light_objects = get_light_objects(bridge, selected_room['lights'])
    
    if not light_objects:
        print("No reachable lights found in this room.")
        return
    
    print(f"✓ Found {len(light_objects)} reachable light(s)")
    
    # Start disco!
    run_disco(bridge, selected_room, light_objects, mode)


if __name__ == "__main__":
    main()

