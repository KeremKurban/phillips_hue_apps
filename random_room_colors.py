"""
Script to randomly change each light's color in a room every 5 seconds.
"""

import colorsys
import random
import time
from phue import Bridge, Group

BRIDGE_IP = "192.168.1.2"
UPDATE_INTERVAL = 5  # seconds


def rgb_to_hue_hsv(r, g, b):
    """
    Convert RGB (0-255) to Hue color space (hue: 0-65535, sat: 0-254, bri: 0-254).
    
    Args:
        r, g, b: RGB values (0-255)
    
    Returns:
        tuple: (hue, saturation, brightness) for Hue lights
    """
    # Normalize RGB to 0-1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    # Convert to HSV (0-1 range)
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    
    # Convert to Hue color space
    hue = int(h * 65535)
    saturation = int(s * 254)
    brightness = int(v * 254)
    
    # Ensure minimum brightness if color is not black
    if brightness == 0 and (r > 0 or g > 0 or b > 0):
        brightness = 1
    
    return hue, saturation, brightness


def get_rooms(bridge):
    """Get all available rooms/groups from the bridge."""
    try:
        groups = bridge.get_group()
        rooms = []
        for group_id, group_info in groups.items():
            # Filter out group 0 which is usually "All Lights"
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


def set_light_random_color(light):
    """Set a light to a random color."""
    try:
        # Generate random RGB values
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Convert to Hue color space
        hue, saturation, brightness = rgb_to_hue_hsv(r, g, b)
        
        # Set the light
        light.on = True
        light.hue = hue
        light.saturation = saturation
        light.brightness = brightness
        
        return r, g, b
    except Exception as e:
        print(f"  Error setting light {light.name}: {e}")
        return None, None, None


def random_room_colors(bridge, room, light_objects):
    """Continuously change each light's color randomly."""
    print(f"\nStarting random color changes for room '{room['name']}'...")
    print(f"Updating every {UPDATE_INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration} - Setting random colors...")
            
            for light in light_objects:
                r, g, b = set_light_random_color(light)
                if r is not None:
                    print(f"  {light.name}: RGB({r}, {g}, {b})")
            
            print(f"\nWaiting {UPDATE_INTERVAL} seconds until next change...\n")
            time.sleep(UPDATE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user.")


def main():
    print("=" * 60)
    print("Hue Random Room Colors")
    print("=" * 60)
    
    # Connect to bridge
    print(f"\nConnecting to Hue Bridge at {BRIDGE_IP}...")
    try:
        bridge = Bridge(BRIDGE_IP)
        bridge.get_api()  # Test connection
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
    
    # Get light objects for the room
    print(f"\nLoading lights in room '{selected_room['name']}'...")
    light_objects = get_light_objects(bridge, selected_room['lights'])
    
    if not light_objects:
        print("No reachable lights found in this room.")
        return
    
    print(f"✓ Found {len(light_objects)} reachable light(s)")
    
    # Start random color changes
    random_room_colors(bridge, selected_room, light_objects)


if __name__ == "__main__":
    main()

