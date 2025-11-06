"""
Helper script to discover Philips Hue Bridge and list available lights.
"""

import sys
from phue import Bridge


def discover_bridge_ip():
    """
    Attempt to discover the Hue Bridge IP.
    Note: This is a simple implementation. For production, consider using
    SSDP/UPnP discovery or the official Hue Bridge discovery API.
    """
    print("=" * 60)
    print("Hue Bridge Discovery")
    print("=" * 60)
    print("\nTo find your Hue Bridge IP address:")
    print("1. Open the Philips Hue app")
    print("2. Go to Settings → Bridge")
    print("3. The IP address will be displayed there")
    print("\nAlternatively, check your router's connected devices list.")
    print("=" * 60 + "\n")


def list_lights(bridge_ip: str):
    """List all lights connected to the Hue Bridge."""
    try:
        print(f"Connecting to bridge at {bridge_ip}...")
        b = Bridge(bridge_ip)
        
        # Try to connect (will prompt for button press if needed)
        try:
            b.get_api()
            print("Connected successfully!")
        except Exception:
            print("\nPlease press the button on your Hue Bridge now...")
            print("Waiting 10 seconds...")
            import time
            time.sleep(10)
            b.connect()
            print("Connected successfully!")
        
        print("\n" + "=" * 60)
        print("Available Lights:")
        print("=" * 60)
        
        lights = b.get_light()
        if not lights:
            print("No lights found!")
            return
        
        for light_id, light_info in lights.items():
            state = "ON" if light_info['state']['on'] else "OFF"
            brightness = light_info['state'].get('bri', 0)
            name = light_info['name']
            
            print(f"\nLight ID: {light_id}")
            print(f"  Name: {name}")
            print(f"  State: {state}")
            print(f"  Brightness: {brightness}/254")
            if 'colormode' in light_info['state']:
                print(f"  Color Mode: {light_info['state']['colormode']}")
        
        print("\n" + "=" * 60)
        print("\nTo control these lights, use:")
        print(f"  python main.py --bridge-ip {bridge_ip} --lights [ID1] [ID2] ...")
        print("\nExample:")
        light_ids = list(lights.keys())[:3]  # Show first 3 lights
        print(f"  python main.py --bridge-ip {bridge_ip} --lights {' '.join(light_ids)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. The bridge IP address is correct")
        print("  2. Your computer and bridge are on the same network")
        print("  3. The bridge is powered on and connected")


def main():
    if len(sys.argv) < 2:
        discover_bridge_ip()
        print("\nUsage: python find_hue_lights.py <bridge_ip>")
        print("Example: python find_hue_lights.py 192.168.1.100")
        sys.exit(1)
    
    bridge_ip = sys.argv[1]
    list_lights(bridge_ip)


if __name__ == "__main__":
    main()

