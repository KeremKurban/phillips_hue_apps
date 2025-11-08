"""
Script to set all lights in a room/group to a specific RGB color.
"""

import colorsys
from phue import Bridge, Group

BRIDGE_IP = "192.168.1.2"


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
    # Hue: 0-65535 (0 = red, 21845 = green, 43690 = blue)
    hue = int(h * 65535)
    # Saturation: 0-254
    saturation = int(s * 254)
    # Brightness: 0-254
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


def get_rgb_input():
    """Get RGB color values from user."""
    print("\nEnter RGB color values (0-255):")
    while True:
        try:
            r = int(input("  Red (0-255): ").strip())
            g = int(input("  Green (0-255): ").strip())
            b = int(input("  Blue (0-255): ").strip())
            
            # Validate range
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                print("Values must be between 0 and 255. Try again.")
                continue
            
            return r, g, b
        except ValueError:
            print("Please enter valid numbers (0-255)")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None, None, None


def set_room_color(bridge, room, r, g, b):
    """Set all lights in a room to the specified RGB color."""
    try:
        # Convert RGB to Hue color space
        hue, saturation, brightness = rgb_to_hue_hsv(r, g, b)
        
        print(f"\nSetting room '{room['name']}' to RGB({r}, {g}, {b})...")
        print(f"  Converted to Hue: H={hue}, S={saturation}, B={brightness}")
        
        # Get the group object
        group = Group(bridge, room['id'])
        
        # Set the color
        group.on = True
        group.hue = hue
        group.saturation = saturation
        group.brightness = brightness
        
        print(f"✓ Successfully set room '{room['name']}' to RGB({r}, {g}, {b})")
        return True
        
    except Exception as e:
        print(f"✗ Error setting room color: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Hue Room Color Control")
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
    
    # Get RGB color
    r, g, b = get_rgb_input()
    if r is None:
        return
    
    # Set the color
    set_room_color(bridge, selected_room, r, g, b)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

