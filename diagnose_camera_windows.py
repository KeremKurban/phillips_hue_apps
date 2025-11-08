"""
Windows Camera Diagnostic Script

This script helps diagnose why a camera might not be accessible in OpenCV
even though it works in other applications like Zoom.
"""

import cv2
import sys

# Define ASCII-safe status symbols for Windows compatibility
if sys.platform.startswith('win'):
    STATUS_OK = "[OK]"
    STATUS_FAIL = "[X]"
else:
    STATUS_OK = "✓"
    STATUS_FAIL = "✗"

def check_windows_camera_permissions():
    """Check if Windows camera permissions are enabled."""
    print("=" * 60)
    print("Windows Camera Permissions Check")
    print("=" * 60)
    print("\nPlease check the following in Windows Settings:")
    print("1. Press Win + I to open Settings")
    print("2. Go to Privacy & Security > Camera")
    print("3. Ensure 'Camera access' is ON")
    print("4. Ensure 'Let apps access your camera' is ON")
    print("5. Ensure 'Let desktop apps access your camera' is ON")
    print("\n" + "=" * 60 + "\n")

def test_camera_with_all_backends(camera_index):
    """Test a camera with all available backends."""
    print(f"\nTesting Camera Index {camera_index}:")
    print("-" * 60)
    
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Any Available"),
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTrying {backend_name}...")
        try:
            cap = cv2.VideoCapture(camera_index, backend_id)
            if cap.isOpened():
                print(f"  {STATUS_OK} Camera opened successfully")
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend_used = cap.getBackendName()
                    
                    print(f"  {STATUS_OK} Frame read successful!")
                    print(f"  Resolution: {width}x{height}")
                    print(f"  FPS: {fps:.2f}")
                    print(f"  Backend used: {backend_used}")
                    cap.release()
                    return True, backend_id, backend_name
                else:
                    print(f"  {STATUS_FAIL} Camera opened but cannot read frames")
                    cap.release()
            else:
                print(f"  {STATUS_FAIL} Could not open camera")
        except Exception as e:
            print(f"  {STATUS_FAIL} Error: {e}")
    
    return False, None, None

def enumerate_cameras_windows():
    """Try to enumerate cameras using cv2-enumerate-cameras."""
    print("\n" + "=" * 60)
    print("Camera Enumeration (cv2-enumerate-cameras)")
    print("=" * 60)
    
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        cameras = enumerate_cameras()
        
        print(f"\nFound {len(cameras)} camera(s):")
        for cam in cameras:
            print(f"\n  Index: {cam.index}")
            print(f"  Name: {cam.name}")
            if hasattr(cam, 'vid') and hasattr(cam, 'pid'):
                print(f"  VID: {cam.vid}, PID: {cam.pid}")
            
            # Test this camera
            works, backend_id, backend_name = test_camera_with_all_backends(cam.index)
            if works:
                print(f"\n  {STATUS_OK} Camera {cam.index} ({cam.name}) WORKS with {backend_name}!")
                return cam.index, backend_id
    except ImportError:
        print("\n  cv2-enumerate-cameras not available")
    except Exception as e:
        print(f"\n  Error: {e}")
    
    return None, None

def test_standard_indices():
    """Test standard camera indices (0-9)."""
    print("\n" + "=" * 60)
    print("Testing Standard Camera Indices (0-9)")
    print("=" * 60)
    
    working_cameras = []
    
    for i in range(10):
        works, backend_id, backend_name = test_camera_with_all_backends(i)
        if works:
            working_cameras.append((i, backend_id, backend_name))
    
    if working_cameras:
        print(f"\n{STATUS_OK} Found {len(working_cameras)} working camera(s):")
        for idx, backend_id, backend_name in working_cameras:
            print(f"  Camera {idx} works with {backend_name}")
        return working_cameras[0][0], working_cameras[0][1]
    else:
        print(f"\n{STATUS_FAIL} No working cameras found in standard indices")
        return None, None

def main():
    print("\n" + "=" * 60)
    print("Windows Camera Diagnostic Tool")
    print("=" * 60)
    
    # Check permissions
    check_windows_camera_permissions()
    
    # Try enumeration first
    cam_idx, backend_id = enumerate_cameras_windows()
    
    # If enumeration didn't work, try standard indices
    if cam_idx is None:
        cam_idx, backend_id = test_standard_indices()
    
    # Final summary
    print("\n" + "=" * 60)
    print("Diagnostic Summary")
    print("=" * 60)
    
    if cam_idx is not None:
        backend_name = "DirectShow" if backend_id == cv2.CAP_DSHOW else \
                      "Media Foundation" if backend_id == cv2.CAP_MSMF else \
                      "Default"
        print(f"\n{STATUS_OK} SUCCESS: Camera found at index {cam_idx}")
        print(f"  Use this command to run your script:")
        print(f"  python main.py --bridge-ip YOUR_BRIDGE_IP --camera {cam_idx}")
        print(f"\n  The camera works with: {backend_name}")
    else:
        print(f"\n{STATUS_FAIL} FAILED: No working cameras found")
        print("\nTroubleshooting steps:")
        print("1. Make sure Zoom and all other camera apps are CLOSED")
        print("2. Check Windows Camera app - does it work there?")
        print("3. Check Device Manager - is the camera listed?")
        print("4. Try unplugging and replugging the USB camera")
        print("5. Check Windows camera privacy settings (see above)")
        print("6. Try restarting your computer")
        print("7. Update camera drivers from manufacturer website")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

