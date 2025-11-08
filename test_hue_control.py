"""
Simple test script to set lights 1-5 to red.
"""

import time
from phue import Bridge

BRIDGE_IP = "192.168.1.2"
LIGHT_IDS = [1, 2, 3, 4, 5]

# Red color in Hue color space (hue value for red is approximately 0 or 65535)
# Hue range: 0-65535 (red is at 0)
RED_HUE = 0
SATURATION = 254  # Full saturation for vivid red
BRIGHTNESS = 254  # Full brightness

def main():
    print(f"Connecting to Hue Bridge at {BRIDGE_IP}...")
    try:
        bridge = Bridge(BRIDGE_IP)
        bridge.get_api()  # Test connection
        print("✓ Connected to bridge")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Press the button on your Hue Bridge and try again...")
        return
    
    print(f"\nSetting lights {LIGHT_IDS} to red...")
    
    for light_id in LIGHT_IDS:
        try:
            light = bridge[light_id]
            print(f"  Light {light_id} ({light.name}): ", end="")
            
            # Turn on, set to red, full brightness and saturation
            light.on = True
            light.hue = RED_HUE
            light.saturation = SATURATION
            light.brightness = BRIGHTNESS
            
            print("✓ Set to red")
            time.sleep(0.2)  # Small delay between lights
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

