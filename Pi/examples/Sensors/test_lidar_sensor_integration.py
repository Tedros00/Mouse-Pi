"""
Test script for LIDAR sensor integration with beam range finder model.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from LidarSensorIntegration import (
    lidar_scan_to_ranges,
    lidar_scan_to_cartesian,
    filter_scan_by_quality,
    filter_scan_by_distance,
    compute_scan_statistics,
    accumulate_scans
)


def generate_mock_scan(num_points: int = 360, 
                       distance_mean: float = 1500,
                       distance_std: float = 300) -> np.ndarray:
    """Generate a mock LIDAR scan for testing."""
    angles = np.linspace(0, 360, num_points, endpoint=False)
    distances = np.random.normal(distance_mean, distance_std, num_points)
    distances = np.clip(distances, 100, 5000)  # Clip to valid range
    quality = np.random.randint(0, 255, num_points)
    
    return np.column_stack([angles, distances, quality])


def test_scan_to_ranges():
    """Test conversion to normalized ranges."""
    print("\n" + "=" * 50)
    print("Test: Scan to Ranges Conversion")
    print("=" * 50)
    
    scan = generate_mock_scan(num_points=360)
    robot_pose = (0, 0, 0)
    
    ranges = lidar_scan_to_ranges(scan, robot_pose, num_beams=360)
    
    assert len(ranges) == 360, "Should have 360 beams"
    assert np.all((ranges >= 0) & (ranges <= 1)), "Ranges should be normalized (0-1)"
    
    print("✓ Correct number of beams: 360")
    print(f"✓ All ranges in [0, 1]: min={np.min(ranges):.3f}, max={np.max(ranges):.3f}")
    print(f"✓ Mean range: {np.mean(ranges):.3f}")
    return True


def test_scan_to_cartesian():
    """Test conversion to cartesian coordinates."""
    print("\n" + "=" * 50)
    print("Test: Scan to Cartesian Conversion")
    print("=" * 50)
    
    scan = generate_mock_scan(num_points=8)  # Simple 8-point scan
    robot_pose = (100, 200, np.pi/4)  # Robot at (100, 200) facing 45 degrees
    
    points = lidar_scan_to_cartesian(scan, robot_pose)
    
    assert points.shape[1] == 2, "Should have x, y coordinates"
    assert len(points) == len(scan), "Should have same number of points"
    
    print(f"✓ Output shape: {points.shape}")
    print(f"✓ X range: {np.min(points[:, 0]):.0f} to {np.max(points[:, 0]):.0f}")
    print(f"✓ Y range: {np.min(points[:, 1]):.0f} to {np.max(points[:, 1]):.0f}")
    return True


def test_filter_by_quality():
    """Test quality filtering."""
    print("\n" + "=" * 50)
    print("Test: Quality Filtering")
    print("=" * 50)
    
    scan = generate_mock_scan(num_points=100)
    original_count = len(scan)
    
    filtered = filter_scan_by_quality(scan, min_quality=100)
    filtered_count = len(filtered)
    
    assert filtered_count <= original_count, "Filtered count should be <= original"
    assert np.all(filtered[:, 2] >= 100), "All quality values should be >= 100"
    
    print(f"✓ Original points: {original_count}")
    print(f"✓ Filtered points (quality >= 100): {filtered_count}")
    print(f"✓ Retention rate: {filtered_count/original_count*100:.1f}%")
    return True


def test_filter_by_distance():
    """Test distance filtering."""
    print("\n" + "=" * 50)
    print("Test: Distance Filtering")
    print("=" * 50)
    
    scan = generate_mock_scan(num_points=100)
    original_count = len(scan)
    
    filtered = filter_scan_by_distance(scan, min_distance=500, max_distance=2500)
    filtered_count = len(filtered)
    
    assert filtered_count <= original_count, "Filtered count should be <= original"
    assert np.all((filtered[:, 1] >= 500) & (filtered[:, 1] <= 2500)), "All distances should be in range"
    
    print(f"✓ Original points: {original_count}")
    print(f"✓ Filtered points (500-2500mm): {filtered_count}")
    print(f"✓ Retention rate: {filtered_count/original_count*100:.1f}%")
    return True


def test_scan_statistics():
    """Test scan statistics computation."""
    print("\n" + "=" * 50)
    print("Test: Scan Statistics")
    print("=" * 50)
    
    scan = generate_mock_scan(num_points=100)
    stats = compute_scan_statistics(scan)
    
    assert 'num_points' in stats, "Should have num_points"
    assert 'min_distance' in stats, "Should have min_distance"
    assert stats['num_points'] == 100, "Should match number of points"
    assert stats['min_distance'] < stats['max_distance'], "Min should be less than max"
    
    print(f"✓ Number of points: {stats['num_points']}")
    print(f"✓ Distance range: {stats['min_distance']:.0f} to {stats['max_distance']:.0f} mm")
    print(f"✓ Mean distance: {stats['mean_distance']:.0f} mm")
    print(f"✓ Quality stats: min={stats['quality_min']:.0f}, max={stats['quality_max']:.0f}")
    return True


def test_accumulate_scans():
    """Test scan accumulation."""
    print("\n" + "=" * 50)
    print("Test: Scan Accumulation")
    print("=" * 50)
    
    scans = [generate_mock_scan(num_points=50) for _ in range(3)]
    poses = [(0, 0, 0), (200, 0, 0), (400, 0, 0)]
    
    accumulated = accumulate_scans(scans, poses, coordinate_frame='world')
    
    assert len(accumulated) == 50 * 3, "Should accumulate all points"
    assert accumulated.shape[1] == 2, "Should have x, y coordinates"
    
    print(f"✓ Total accumulated points: {len(accumulated)}")
    print(f"✓ X range: {np.min(accumulated[:, 0]):.0f} to {np.max(accumulated[:, 0]):.0f} mm")
    print(f"✓ Y range: {np.min(accumulated[:, 1]):.0f} to {np.max(accumulated[:, 1]):.0f} mm")
    return True


def test_empty_scan():
    """Test handling of empty scans."""
    print("\n" + "=" * 50)
    print("Test: Empty Scan Handling")
    print("=" * 50)
    
    empty_scan = np.array([]).reshape(0, 3)
    robot_pose = (0, 0, 0)
    
    ranges = lidar_scan_to_ranges(empty_scan, robot_pose)
    stats = compute_scan_statistics(empty_scan)
    
    assert len(ranges) == 360, "Should return full-range values"
    assert np.all(ranges == 1.0), "Should return max range"
    assert stats['num_points'] == 0, "Should show 0 points"
    
    print("✓ Empty scan returns max-range values")
    print("✓ Statistics show 0 points")
    return True


def main():
    print("\n╔" + "=" * 48 + "╗")
    print("║" + " " * 8 + "LIDAR Sensor Integration Tests" + " " * 10 + "║")
    print("╚" + "=" * 48 + "╝")
    
    tests = [
        ("Scan to Ranges", test_scan_to_ranges),
        ("Scan to Cartesian", test_scan_to_cartesian),
        ("Quality Filtering", test_filter_by_quality),
        ("Distance Filtering", test_filter_by_distance),
        ("Scan Statistics", test_scan_statistics),
        ("Scan Accumulation", test_accumulate_scans),
        ("Empty Scan Handling", test_empty_scan),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"✗ {name} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 50 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
