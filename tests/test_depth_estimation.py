"""Unit tests for the depth estimation module."""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.depth_estimation.utils import normalize_depth_map, convert_to_uint8


class TestDepthUtils(unittest.TestCase):
    """Test cases for depth estimation utility functions."""
    
    def test_normalize_depth_map(self) -> None:
        """Test the normalize_depth_map function."""
        # Create a test depth map
        depth_map = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Test normalization
        normalized = normalize_depth_map(depth_map)
        
        # Check results
        self.assertEqual(normalized.shape, depth_map.shape)
        self.assertAlmostEqual(normalized.min(), 0.0)
        self.assertAlmostEqual(normalized.max(), 1.0)
        
        # Test with custom min/max values
        custom_normalized = normalize_depth_map(depth_map, min_val=0.0, max_val=10.0)
        self.assertEqual(custom_normalized.shape, depth_map.shape)
        self.assertAlmostEqual(custom_normalized.min(), 0.1)
        self.assertAlmostEqual(custom_normalized.max(), 0.6)
        
        # Test with equal min/max (edge case)
        constant_depth = np.ones((3, 3), dtype=np.float32)
        constant_normalized = normalize_depth_map(constant_depth)
        self.assertTrue(np.all(constant_normalized == 0.0))
    
    def test_convert_to_uint8(self) -> None:
        """Test the convert_to_uint8 function."""
        # Create a test depth map (already normalized)
        depth_map = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        
        # Convert to uint8
        uint8_depth = convert_to_uint8(depth_map, normalize=False)
        
        # Check results
        self.assertEqual(uint8_depth.dtype, np.uint8)
        self.assertEqual(uint8_depth.shape, depth_map.shape)
        self.assertEqual(uint8_depth[0, 0], 0)
        self.assertEqual(uint8_depth[0, 1], 127)
        self.assertEqual(uint8_depth[0, 2], 255)
        
        # Test with normalization
        unnormalized = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        uint8_unnormalized = convert_to_uint8(unnormalized, normalize=True)
        self.assertEqual(uint8_unnormalized.dtype, np.uint8)
        self.assertEqual(uint8_unnormalized[0, 0], 0)
        self.assertEqual(uint8_unnormalized[0, 2], 255)


if __name__ == "__main__":
    unittest.main() 