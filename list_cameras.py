"""
Helper script to list all available cameras on the system.

This is useful for identifying which camera index corresponds to
your Meta Quest 2 or other camera devices.
"""

import cv2
import sys
import os

# Define ASCII-safe status symbols for Windows compatibility
if sys.platform.startswith('win'):
    STATUS_OK = "[OK]"
    STATUS_FAIL = "[X]"
    STATUS_TIP = "TIP:"
else:
    STATUS_OK = "✓"
    STATUS_FAIL = "✗"
    STATUS_TIP = "💡"

# Try to import cv2-enumerate-cameras for better camera enumeration
try:
    from cv2_enumerate_cameras import enumerate_cameras
    HAS_ENUMERATE_CAMERAS = True
except ImportError:
    HAS_ENUMERATE_CAMERAS = False


def try_camera_with_backend(camera_index, backend=None, timeout_ms=5000):
    """Try to open a camera with a specific backend."""
    if backend is not None:
        cap = cv2.VideoCapture(camera_index, backend)
    else:
        cap = cv2.VideoCapture(camera_index)
    
    if cap.isOpened():
        # Set timeout for frame reading (important for USB passthrough)
        if sys.platform.startswith('linux'):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid stale frames
        
        # Try to read a frame to verify it's actually working
        # Give it a few tries with small delays
        ret = False
        frame = None
        for attempt in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            import time
            time.sleep(0.1)  # Small delay between attempts
        
        if ret and frame is not None:
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend_name = cap.getBackendName()
            
            cap.release()
            return {
                'working': True,
                'width': width,
                'height': height,
                'fps': fps,
                'backend': backend_name,
                'backend_id': backend
            }
        cap.release()
    
    return {'working': False}


def list_cameras_with_enumerate():
    """List cameras using cv2-enumerate-cameras library."""
    print("\nUsing cv2-enumerate-cameras library for better detection...")
    print("-" * 60)
    
    cameras = enumerate_cameras()
    available_cameras = []
    
    for cam in cameras:
        print(f"\nCamera Index {cam.index}:")
        print(f"  Name: {cam.name}")
        if hasattr(cam, 'vid') and hasattr(cam, 'pid'):
            print(f"  VID: {cam.vid}, PID: {cam.pid}")
        
        # Try to actually open and test the camera
        # On Linux/WSL, try V4L2 first, then default
        working_idx = cam.index
        if sys.platform.startswith('linux'):
            result = try_camera_with_backend(cam.index, cv2.CAP_V4L2)
            if not result['working']:
                result = try_camera_with_backend(cam.index, None)
            
            # If the index is unusual (like 1800, 200), also try standard indices
            # Sometimes cv2-enumerate-cameras reports wrong indices
            if not result['working'] and cam.index > 10:
                print(f"  Note: Unusual index detected, trying standard indices...")
                for std_idx in [0, 1, 2]:
                    result = try_camera_with_backend(std_idx, cv2.CAP_V4L2)
                    if result['working']:
                        print(f"  {STATUS_OK} Camera works at index {std_idx} instead of {cam.index}")
                        working_idx = std_idx
                        break
        else:
            # On Windows, try DirectShow, Media Foundation, then default
            result = try_camera_with_backend(cam.index, cv2.CAP_DSHOW)
            if not result['working']:
                result = try_camera_with_backend(cam.index, cv2.CAP_MSMF)
            if not result['working']:
                result = try_camera_with_backend(cam.index, None)
        
        if result['working']:
            print(f"  Resolution: {result['width']}x{result['height']}")
            print(f"  FPS: {result['fps']:.2f}")
            print(f"  Backend: {result['backend']}")
            print(f"  Status: {STATUS_OK} Working")
            available_cameras.append(working_idx)
        else:
            print(f"  Status: {STATUS_FAIL} Detected but cannot read frames")
            print(f"  {STATUS_TIP} Try using index 0 or 1 manually: --camera 0 or --camera 1")
    
    return available_cameras


def list_cameras():
    """List all available cameras and their indices."""
    print("=" * 60)
    print("Available Cameras:")
    print("=" * 60)
    
    available_cameras = []
    
    # First, try using cv2-enumerate-cameras if available
    if HAS_ENUMERATE_CAMERAS:
        try:
            available_cameras = list_cameras_with_enumerate()
            if available_cameras:
                print("\n" + "=" * 60)
                print(f"Found {len(available_cameras)} working camera(s): {available_cameras}")
                print("\nTo use a camera, specify its index with --camera:")
                print(f"  python main.py --bridge-ip YOUR_BRIDGE_IP --camera {available_cameras[0]}")
                print("=" * 60)
                return
        except Exception as e:
            print(f"\nWarning: cv2-enumerate-cameras failed: {e}")
            print("Falling back to manual enumeration...\n")
    
    # Fallback: Manual enumeration with multiple backends
    print("\nManually checking camera indices...")
    print("-" * 60)
    
    # On Linux/WSL, also try direct device paths
    if sys.platform.startswith('linux'):
        print("\nTrying direct device paths (/dev/video*)...")
        import os
        for dev_path in ['/dev/video0', '/dev/video1', '/dev/video2']:
            if os.path.exists(dev_path):
                print(f"\nTrying {dev_path}...")
                # Try opening by device path string
                cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        backend = cap.getBackendName()
                        
                        # Find the index for this device
                        device_index = int(dev_path.replace('/dev/video', ''))
                        
                        print(f"  {STATUS_OK} Found working camera at {dev_path}")
                        print(f"  Resolution: {width}x{height}")
                        print(f"  FPS: {fps:.2f}")
                        print(f"  Backend: {backend}")
                        print(f"  Use index: {device_index}")
                        available_cameras.append(device_index)
                    cap.release()
    
    # Define backends to try (Windows-specific)
    backends_to_try = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (None, "Default"),
    ]
    
    # Also try V4L2 on Linux
    if sys.platform.startswith('linux'):
        backends_to_try.insert(0, (cv2.CAP_V4L2, "V4L2"))
    
    max_cameras_to_check = 10  # Check up to 10 camera indices
    
    for i in range(max_cameras_to_check):
        camera_found = False
        
        for backend_id, backend_name in backends_to_try:
            result = try_camera_with_backend(i, backend_id)
            
            if result['working']:
                if not camera_found:  # Only print once per camera index
                    print(f"\nCamera {i}:")
                    print(f"  Resolution: {result['width']}x{result['height']}")
                    print(f"  FPS: {result['fps']:.2f}")
                    print(f"  Backend: {result['backend']} ({backend_name})")
                    print(f"  Status: {STATUS_OK} Working")
                    available_cameras.append(i)
                    camera_found = True
                    break  # Found working camera, no need to try other backends
        
        if not camera_found:
            print(f"\nCamera {i}: {STATUS_FAIL} Not available")
    
    print("\n" + "=" * 60)
    if available_cameras:
        print(f"Found {len(available_cameras)} working camera(s): {available_cameras}")
        print("\nTo use a camera, specify its index with --camera:")
        print(f"  python main.py --bridge-ip YOUR_BRIDGE_IP --camera {available_cameras[0]}")
    else:
        print("No working cameras found!")
        print("\nTroubleshooting:")
        print("  - Make sure your camera is connected")
        print("  - Close Zoom or any other application using the camera")
        print("  - On Windows, try disconnecting and reconnecting the USB camera")
        print("  - Check Device Manager to ensure the camera is recognized")
        print("  - On Linux, you may need to grant camera permissions")
        print("  - For Quest 2, ensure it's connected via Quest Link/Air Link")
        if sys.platform.startswith('linux'):
            print("\nWSL USB Passthrough Notes:")
            print("  - Make sure camera is attached: usbipd attach --wsl --busid <BUSID>")
            print("  - Try using index 0 or 1 directly: --camera 0 or --camera 1")
            print("  - Check /dev/video* devices exist: ls -la /dev/video*")
            print("  - Ensure you're in the 'video' group: groups (should include 'video')")
        print("\nNote: If your camera works in Zoom but not here, it might be:")
        print("  - Locked by another application (close Zoom completely)")
        print("  - Using a different API that OpenCV doesn't access by default")
        if not HAS_ENUMERATE_CAMERAS:
            print(f"\n{STATUS_TIP} TIP: Install cv2-enumerate-cameras for better camera detection:")
            print("     pip install cv2-enumerate-cameras")
    print("=" * 60)


if __name__ == "__main__":
    list_cameras()

