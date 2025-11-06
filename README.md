# Hue Hand Control

Control your Philips Hue lights using hand gestures! This application tracks your thumb and index finger tips using MediaPipe and OpenCV, calculates the distance between them, and uses that distance to control the brightness of your Philips Hue lights.

## Features

- 🖐️ Real-time hand tracking using MediaPipe
- 💡 Control Philips Hue light brightness with finger gestures
- 📏 Automatic distance-to-brightness mapping
- 🎯 Support for controlling multiple lights simultaneously
- 🔄 Dynamic calibration reset

## Requirements

- Python 3.8 or higher
- A Philips Hue Bridge (connected to your network)
- A webcam/camera (or Meta Quest 2 with Quest Link/Air Link)
- Python packages (see `requirements.txt`)

## Installation

1. Clone or download this repository

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Find Your Philips Hue Bridge IP Address

You can find your Hue Bridge IP address by:
- Using the Philips Hue app (Settings → Bridge)
- Checking your router's connected devices
- Using a network scanner tool

### 2. Connect to the Hue Bridge

On first run, the application will prompt you to press the button on your Hue Bridge. This creates an authentication token that is saved for future use.

### 3. Find Your Light IDs

You can find your light IDs by:
- Using the Philips Hue app
- Running this Python code:

```python
from phue import Bridge

b = Bridge('YOUR_BRIDGE_IP')
b.connect()
lights = b.get_light()
for light_id, light_info in lights.items():
    print(f"Light {light_id}: {light_info['name']}")
```

## Usage

### Basic Usage

```bash
python main.py --bridge-ip YOUR_BRIDGE_IP
```

This will control light ID 1 (default) using default distance ranges (20-200 pixels).

### Control Multiple Lights

```bash
python main.py --bridge-ip 192.168.1.100 --lights 1 2 3
```

### Customize Distance Ranges

```bash
python main.py --bridge-ip 192.168.1.100 --min-distance 30 --max-distance 300
```

### Use a Different Camera

```bash
python main.py --bridge-ip 192.168.1.100 --camera 1
```

### List Available Cameras

```bash
python list_cameras.py
```

This helper script lists all available cameras and their indices, which is useful for identifying your Quest 2 or other camera devices.

### Full Example

```bash
python main.py \
  --bridge-ip 192.168.1.100 \
  --lights 1 2 3 \
  --min-distance 25 \
  --max-distance 250 \
  --camera 0
```

## Meta Quest 2 Usage

The application includes a dedicated Quest 2 version that's optimized for VR passthrough camera input.

### Quest 2 Setup

1. **Connect Quest 2 to PC:**
   - **Option A (Quest Link)**: Connect via USB-C cable and enable Quest Link in the headset
   - **Option B (Air Link)**: Enable Air Link in both the Oculus PC app and Quest 2 headset (requires 5GHz Wi-Fi)

2. **Enable Passthrough (Optional but Recommended):**
   - Enable passthrough mode in Quest 2 to see your hands in VR
   - This provides better visual feedback while controlling lights

3. **Find Quest 2 Camera Index:**
   ```bash
   python list_cameras.py
   ```
   This will list all available cameras. Quest 2 typically appears as camera index 1, 2, or higher.

### Quest 2 Basic Usage

```bash
python main_quest2.py --bridge-ip YOUR_BRIDGE_IP
```

The script will auto-detect the Quest 2 camera if available.

### Quest 2 with Specific Camera

```bash
python main_quest2.py --bridge-ip 192.168.1.100 --camera 1
```

### Quest 2 Control Multiple Lights

```bash
python main_quest2.py --bridge-ip 192.168.1.100 --lights "Living Room" "Bedroom lamp"
```

### Quest 2 List Available Cameras

```bash
python main_quest2.py --list-cameras
```

### Quest 2 Tips

- **Use Passthrough Mode**: Enable passthrough in Quest 2 to see your hands and the camera feed
- **Positioning**: Position your hands comfortably in front of the headset (not too close or too far)
- **Lighting**: Good lighting improves hand tracking accuracy, even in VR
- **Calibration**: The Quest 2 camera may have different resolution, so you might need to adjust `--min-distance` and `--max-distance` values
- **Visual Feedback**: The Quest 2 version includes enhanced visual indicators (larger circles, thicker lines) for better visibility in VR

### Quest 2 Troubleshooting

**Quest 2 Camera Not Detected:**
- Ensure Quest Link or Air Link is active and connected
- Run `python list_cameras.py` to see available cameras
- Try different camera indices: `--camera 1`, `--camera 2`, etc.
- Make sure no other application is using the Quest 2 camera

**Quest 2 Connection Issues:**
- Verify Quest Link/Air Link is working properly
- Check that the Oculus PC app is running
- For Air Link, ensure both devices are on the same 5GHz Wi-Fi network
- Try restarting Quest Link/Air Link connection

**Quest 2 Hand Tracking Issues:**
- Ensure good lighting in your physical environment
- Position your hands clearly in front of the headset
- Avoid covering hands with objects or shadows
- The Quest 2 passthrough camera may have different tracking characteristics than a regular webcam

## How It Works

1. **Hand Detection**: MediaPipe detects your hand and identifies 21 hand landmarks
2. **Fingertip Tracking**: The application tracks the thumb tip (landmark 4) and index finger tip (landmark 8)
3. **Distance Calculation**: Calculates the Euclidean distance between the two fingertips
4. **Brightness Mapping**: Maps the distance to a brightness value (1-254) using the specified min/max range
5. **Light Control**: Updates the brightness of the specified Philips Hue lights

## Controls

- **Move fingers closer**: Decreases brightness (dimmer)
- **Move fingers farther**: Increases brightness (brighter)
- **Press 'q'**: Quit the application
- **Press 'r'**: Reset calibration (recalculates min/max based on observed distances)

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--bridge-ip` | IP address of Hue Bridge | Required |
| `--lights` | List of light IDs to control | `[1]` |
| `--min-distance` | Minimum distance (pixels) for min brightness | `20.0` |
| `--max-distance` | Maximum distance (pixels) for max brightness | `200.0` |
| `--camera` | Camera index | `0` |

## Troubleshooting

### Camera Not Working

- Make sure your camera is connected and not being used by another application
- Try different camera indices: `--camera 0`, `--camera 1`, etc.
- On Linux, you may need to grant camera permissions
- Use `python list_cameras.py` to see available cameras and their indices

### Hue Bridge Connection Issues

- Ensure your Hue Bridge and computer are on the same network
- Press the button on the Hue Bridge when prompted
- Check that the IP address is correct
- Verify the bridge is powered on and connected

### Hand Not Detected

- Ensure good lighting conditions
- Make sure your hand is clearly visible to the camera
- Try moving your hand closer to the camera
- Ensure your hand is not obscured by objects

### Lights Not Responding

- Verify the light IDs are correct
- Check that the lights are powered on
- Ensure the lights are connected to the Hue Bridge
- Try controlling lights individually to identify issues

## Tips for Best Results

1. **Lighting**: Use good, even lighting for better hand detection
2. **Background**: Use a plain background for better contrast
3. **Distance**: Keep your hand at a comfortable distance from the camera (not too close or too far)
4. **Calibration**: Use the 'r' key to reset calibration if the brightness range doesn't feel right
5. **Stability**: Keep your hand steady for smoother brightness changes

## License

This project is provided as-is for educational and personal use.

## Scripts

- **`main.py`**: Standard camera-based hand tracking (works with any webcam)
- **`main_quest2.py`**: Optimized for Meta Quest 2 camera input
- **`find_hue_lights.py`**: Helper script to list available Hue lights
- **`list_cameras.py`**: Helper script to list available cameras and their indices

## Credits

- Uses [MediaPipe](https://mediapipe.dev/) for hand tracking
- Uses [OpenCV](https://opencv.org/) for video processing
- Uses [phue](https://github.com/studioimaginaire/phue) for Philips Hue control

