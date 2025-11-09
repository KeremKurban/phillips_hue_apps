"""
Fetch light positions and information from Philips Hue Bridge.
Explores all available data from lights in a room, including position information
if available from the Entertainment Areas or light placement features.
"""

import json
from phue import Bridge
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

BRIDGE_IP = "192.168.1.2"


def get_rooms(bridge: Bridge) -> List[Dict]:
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


def display_rooms(rooms: List[Dict]):
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


def select_room(rooms: List[Dict]) -> Optional[Dict]:
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


def get_raw_api_data(bridge: Bridge, endpoint: str) -> Optional[Dict]:
    """Get raw API data from a specific endpoint."""
    try:
        # Access the raw API through the bridge's request method
        response = bridge.request('GET', endpoint)
        return response
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        return None


def get_light_details(bridge: Bridge, light_id: str) -> Dict:
    """Get all available details for a specific light."""
    details = {
        'id': light_id,
        'name': None,
        'state': None,
        'config': None,
        'capabilities': None,
        'swupdate': None,
        'type': None,
        'modelid': None,
        'manufacturername': None,
        'productname': None,
        'swversion': None,
        'uniqueid': None,
        'position': None,
        'raw_data': None
    }
    
    try:
        # Get light object
        light = bridge[int(light_id)]
        details['name'] = light.name
        
        # Get full light info from API
        lights_data = bridge.get_light()
        if light_id in lights_data:
            light_info = lights_data[light_id]
            details['state'] = light_info.get('state', {})
            details['config'] = light_info.get('config', {})
            details['capabilities'] = light_info.get('capabilities', {})
            details['swupdate'] = light_info.get('swupdate', {})
            details['type'] = light_info.get('type', '')
            details['modelid'] = light_info.get('modelid', '')
            details['manufacturername'] = light_info.get('manufacturername', '')
            details['productname'] = light_info.get('productname', '')
            details['swversion'] = light_info.get('swversion', '')
            details['uniqueid'] = light_info.get('uniqueid', '')
            
            # Store raw data for inspection
            details['raw_data'] = light_info
            
            # Check for position data in various places
            # Position might be in config, state, or at the root level
            if 'position' in light_info:
                details['position'] = light_info['position']
            elif 'config' in light_info and 'position' in light_info['config']:
                details['position'] = light_info['config']['position']
            elif 'state' in light_info and 'position' in light_info['state']:
                details['position'] = light_info['state']['position']
        
        # Try to get Entertainment Area data if available
        # Entertainment areas might have position data
        try:
            entertainment_data = get_raw_api_data(bridge, '/entertainment')
            if entertainment_data:
                details['entertainment_data'] = entertainment_data
        except:
            pass
            
    except Exception as e:
        print(f"Error getting details for light {light_id}: {e}")
    
    return details


def get_group_details(bridge: Bridge, group_id: str) -> Dict:
    """Get all available details for a group/room."""
    details = {
        'id': group_id,
        'name': None,
        'lights': [],
        'state': None,
        'action': None,
        'class': None,
        'positions': None,
        'raw_data': None
    }
    
    try:
        groups_data = bridge.get_group()
        if group_id in groups_data:
            group_info = groups_data[group_id]
            details['name'] = group_info.get('name', '')
            details['lights'] = group_info.get('lights', [])
            details['state'] = group_info.get('state', {})
            details['action'] = group_info.get('action', {})
            details['class'] = group_info.get('class', '')
            details['raw_data'] = group_info
            
            # Check for position data in group
            if 'positions' in group_info:
                details['positions'] = group_info['positions']
            elif 'locations' in group_info:
                details['positions'] = group_info['locations']
            elif 'light_positions' in group_info:
                details['positions'] = group_info['light_positions']
    except Exception as e:
        print(f"Error getting details for group {group_id}: {e}")
    
    return details


def display_light_info(light_details: Dict, show_full_details: bool = False):
    """Display formatted light information."""
    print(f"\n{'='*60}")
    print(f"Light ID: {light_details['id']}")
    print(f"Name: {light_details['name']}")
    print(f"{'='*60}")
    
    if show_full_details:
        if light_details['type']:
            print(f"Type: {light_details['type']}")
        if light_details['modelid']:
            print(f"Model: {light_details['modelid']}")
        if light_details['manufacturername']:
            print(f"Manufacturer: {light_details['manufacturername']}")
        if light_details['productname']:
            print(f"Product: {light_details['productname']}")
        if light_details['swversion']:
            print(f"Software Version: {light_details['swversion']}")
        if light_details['uniqueid']:
            print(f"Unique ID: {light_details['uniqueid']}")
        
        # Display state information
        if light_details['state']:
            state = light_details['state']
            print(f"\nState:")
            print(f"  On: {state.get('on', 'N/A')}")
            print(f"  Brightness: {state.get('bri', 'N/A')}/254")
            print(f"  Reachable: {state.get('reachable', 'N/A')}")
            if 'xy' in state:
                print(f"  Color (xy): {state['xy']}")
            if 'hue' in state:
                print(f"  Hue: {state['hue']}")
            if 'sat' in state:
                print(f"  Saturation: {state['sat']}")
            if 'ct' in state:
                print(f"  Color Temperature: {state['ct']}")
    
    # Note: Position data is typically at room level, not individual light level
    if light_details['position']:
        print(f"\n{'='*60}")
        print("POSITION DATA FOUND:")
        print(f"{'='*60}")
        print(json.dumps(light_details['position'], indent=2))


def plot_light_positions_3d(group_details: Dict, lights_data: List[Dict], room_name: str):
    """
    Create a 3D visualization of light positions in the room.
    
    Args:
        group_details: Room/group details containing positions
        lights_data: List of light details with names
        room_name: Name of the room for the plot title
    """
    if not group_details.get('positions'):
        print("\n⚠ No position data available for 3D visualization.")
        return
    
    # Create a mapping of light ID to light name
    light_id_to_name = {light['id']: light['name'] for light in lights_data}
    
    # Extract positions
    positions = group_details['positions']
    light_ids = list(positions.keys())
    
    # Extract coordinates
    x_coords = [positions[light_id][0] for light_id in light_ids]
    y_coords = [positions[light_id][1] for light_id in light_ids]
    z_coords = [positions[light_id][2] for light_id in light_ids]
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the lights
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        s=200, c=range(len(light_ids)), 
                        cmap='tab20', alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels for each light
    for i, light_id in enumerate(light_ids):
        light_name = light_id_to_name.get(light_id, f"Light {light_id}")
        # Truncate long names for readability
        if len(light_name) > 20:
            light_name = light_name[:17] + "..."
        ax.text(x_coords[i], y_coords[i], z_coords[i], 
               f"  {light_id}\n  {light_name}", 
               fontsize=8, ha='left')
    
    # Set labels and title
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position', fontsize=12, fontweight='bold')
    ax.set_title(f'3D Light Positions - {room_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([max(x_coords) - min(x_coords),
                          max(y_coords) - min(y_coords),
                          max(z_coords) - min(z_coords)]).max() / 2.0
    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add a legend with light IDs and names
    legend_elements = []
    for light_id in light_ids:
        light_name = light_id_to_name.get(light_id, f"Light {light_id}")
        legend_elements.append(f"ID {light_id}: {light_name}")
    
    # Create a text box with legend (since 3D plots don't support traditional legends well)
    legend_text = "\n".join(legend_elements)
    ax.text2D(0.02, 0.98, legend_text, transform=ax.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add coordinate info
    info_text = f"Total lights: {len(light_ids)}\n"
    info_text += f"X range: [{min(x_coords):.2f}, {max(x_coords):.2f}]\n"
    info_text += f"Y range: [{min(y_coords):.2f}, {max(y_coords):.2f}]\n"
    info_text += f"Z range: [{min(z_coords):.2f}, {max(z_coords):.2f}]"
    ax.text2D(0.98, 0.02, info_text, transform=ax.transAxes,
             fontsize=8, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✓ 3D visualization displayed!")
    print("  - Each light is marked with its ID and name")
    print("  - Colors differentiate between lights")
    print("  - Check if the positions match your room layout")


def display_group_info(group_details: Dict):
    """Display formatted group/room information."""
    print(f"\n{'='*60}")
    print(f"Room/Group: {group_details['name']} (ID: {group_details['id']})")
    print(f"{'='*60}")
    print(f"Lights: {', '.join(group_details['lights'])}")
    
    if group_details['class']:
        print(f"Class: {group_details['class']}")
    
    # Display position information if found
    if group_details['positions']:
        print(f"\n{'='*60}")
        print("✓ POSITION DATA FOUND IN ROOM/GROUP!")
        print(f"{'='*60}")
        print("Light positions (X, Y, Z coordinates):")
        for light_id, pos in group_details['positions'].items():
            print(f"  Light ID {light_id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # Display raw data for inspection (optional, commented out to reduce clutter)
    # if group_details['raw_data']:
    #     print(f"\n{'='*60}")
    #     print("Raw Group Data (for inspection):")
    #     print(f"{'='*60}")
    #     print(json.dumps(group_details['raw_data'], indent=2))


def main():
    print("=" * 60)
    print("Philips Hue Light Position Fetcher")
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
    
    print(f"\n{'='*60}")
    print(f"Fetching information for room: {selected_room['name']}")
    print(f"{'='*60}")
    
    # Get group details
    print("\nFetching group/room details...")
    group_details = get_group_details(bridge, selected_room['id'])
    display_group_info(group_details)
    
    # Get details for each light in the room
    print(f"\n{'='*60}")
    print(f"Fetching details for {len(selected_room['lights'])} light(s)...")
    print(f"{'='*60}")
    
    all_lights_data = []
    for light_id in selected_room['lights']:
        print(f"Processing light {light_id}...", end=' ')
        light_details = get_light_details(bridge, light_id)
        all_lights_data.append(light_details)
        print(f"✓ {light_details['name']}")
    
    # Summary with room-level positions
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Room: {selected_room['name']}")
    print(f"Total Lights: {len(all_lights_data)}")
    
    # Check for room-level positions
    if group_details.get('positions'):
        print(f"\n✓ Room-level position data found for {len(group_details['positions'])} light(s)!")
        print("\nLight positions (X, Y, Z):")
        for light_id, pos in group_details['positions'].items():
            light_name = next((l['name'] for l in all_lights_data if l['id'] == light_id), f"Light {light_id}")
            print(f"  {light_name} (ID: {light_id}): [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    else:
        print("\n⚠ No room-level position data found.")
        print("Position data might be stored in Entertainment Areas or")
        print("may not be accessible through the standard API.")
    
    # Check for individual light positions (rare)
    lights_with_positions = [l for l in all_lights_data if l['position']]
    if lights_with_positions:
        print(f"\nIndividual light position data found for {len(lights_with_positions)} light(s):")
        for light in lights_with_positions:
            print(f"  - {light['name']} (ID: {light['id']}): {light['position']}")
    
    # Option to visualize
    if group_details.get('positions'):
        visualize = input("\nCreate 3D visualization of light positions? (y/n): ").strip().lower()
        if visualize == 'y':
            plot_light_positions_3d(group_details, all_lights_data, selected_room['name'])
    
    # Option to save to JSON
    save = input("\nSave all data to JSON file? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"hue_room_data_{selected_room['name'].replace(' ', '_')}.json"
        output = {
            'room': group_details,
            'lights': all_lights_data
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Data saved to {filename}")


if __name__ == "__main__":
    main()

