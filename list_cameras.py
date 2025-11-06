"""
Helper script to list all available cameras on the system.

This is useful for identifying which camera index corresponds to
your Meta Quest 2 or other camera devices.
"""

import cv2


def list_cameras():
    """List all available cameras and their indices."""
    print("=" * 60)
    print("Available Cameras:")
    print("=" * 60)
    
    available_cameras = []
    max_cameras_to_check = 10  # Check up to 10 camera indices
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it's actually working
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to get backend name (may not work on all systems)
                backend = cap.getBackendName()
                
                print(f"\nCamera {i}:")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps:.2f}")
                print(f"  Backend: {backend}")
                print(f"  Status: ✓ Working")
                
                available_cameras.append(i)
            else:
                print(f"\nCamera {i}: ✗ Opened but cannot read frames")
            cap.release()
        else:
            print(f"\nCamera {i}: ✗ Not available")
    
    print("\n" + "=" * 60)
    if available_cameras:
        print(f"Found {len(available_cameras)} working camera(s): {available_cameras}")
        print("\nTo use a camera, specify its index with --camera:")
        print(f"  python main.py --bridge-ip YOUR_BRIDGE_IP --camera {available_cameras[0]}")
    else:
        print("No working cameras found!")
        print("\nTroubleshooting:")
        print("  - Make sure your camera is connected")
        print("  - Check if another application is using the camera")
        print("  - On Linux, you may need to grant camera permissions")
        print("  - For Quest 2, ensure it's connected via Quest Link/Air Link")
    print("=" * 60)


if __name__ == "__main__":
    list_cameras()

