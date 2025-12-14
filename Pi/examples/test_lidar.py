"""
Test script to validate RPlidar interface functions.
"""

import sys
sys.path.insert(0, '/home/mouse/AMR/src')

from lidar import connect_lidar, init_lidar, capture_map, disconnect_lidar
import numpy as np

def test_connect():
    """Test LIDAR connection."""
    print("=" * 50)
    print("Testing LIDAR Connection")
    print("=" * 50)
    try:
        lidar = connect_lidar(port='/dev/ttyUSB1')
        print("✓ Connection test passed")
        return lidar
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return None


def test_init(lidar):
    """Test LIDAR initialization."""
    print("\n" + "=" * 50)
    print("Testing LIDAR Initialization")
    print("=" * 50)
    try:
        init_lidar(lidar)
        print("✓ Initialization test passed")
        return True
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        return False


def test_capture(lidar):
    """Test LIDAR scan capture."""
    print("\n" + "=" * 50)
    print("Testing LIDAR Scan Capture")
    print("=" * 50)
    try:
        # Capture with max_points limit for faster testing
        scan_data = capture_map(lidar, max_distance=5000, max_points=1000)
        
        # Validate returned data
        assert isinstance(scan_data, np.ndarray), "Scan data should be a NumPy array"
        assert scan_data.shape[1] == 3, "Each point should have 3 values (angle, distance, quality)"
        assert len(scan_data) > 0, "Scan should contain at least one point"
        
        print(f"✓ Captured {len(scan_data)} points")
        print(f"  - Angle range: {scan_data[:, 0].min():.1f}° to {scan_data[:, 0].max():.1f}°")
        print(f"  - Distance range: {scan_data[:, 1].min():.0f}mm to {scan_data[:, 1].max():.0f}mm")
        print(f"  - Quality range: {scan_data[:, 2].min():.0f} to {scan_data[:, 2].max():.0f}")
        print("✓ Capture test passed")
        
        return True
    except Exception as e:
        print(f"✗ Capture test failed: {e}")
        return False


def test_disconnect(lidar):
    """Test LIDAR disconnection."""
    print("\n" + "=" * 50)
    print("Testing LIDAR Disconnection")
    print("=" * 50)
    try:
        disconnect_lidar(lidar)
        print("✓ Disconnection test passed")
        return True
    except Exception as e:
        print(f"✗ Disconnection test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 10 + "RPlidar Interface Validation Tests" + " " * 4 + "║")
    print("╚" + "=" * 48 + "╝")
    
    # Run tests in sequence
    lidar = test_connect()
    if not lidar:
        print("\n✗ Cannot proceed without LIDAR connection")
        return
    
    if not test_init(lidar):
        print("\n✗ Cannot proceed without LIDAR initialization")
        disconnect_lidar(lidar)
        return
    
    test_capture(lidar)
    test_disconnect(lidar)
    
    print("\n" + "=" * 50)
    print("All tests completed")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
