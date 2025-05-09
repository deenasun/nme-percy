"""Unit tests for the grid conversion module."""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.grid_conversion.converter import depth_to_grid, add_start_goal_points, FREE_SPACE, OBSTACLE, START_POINT, GOAL_POINT


class TestGridConversion(unittest.TestCase):
    """Test cases for grid conversion functionality."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a simple test depth map
        self.depth_map = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.6],
            [0.5, 0.4, 0.3, 0.2]
        ], dtype=np.float32)
    
    def test_depth_to_grid(self) -> None:
        """Test the depth_to_grid function."""
        # Convert to grid with default parameters
        grid = depth_to_grid(
            depth_map=self.depth_map,
            grid_size=4,  # Match the input size
            threshold_factor=0.5,
            smoothing=False  # Disable smoothing for deterministic testing
        )
        
        # Check output shape
        self.assertEqual(grid.shape, (4, 4))
        
        # Check data type
        self.assertEqual(grid.dtype, np.int8)
        
        # Check thresholding (values < 0.5 should be obstacles)
        expected_grid = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 1]
        ], dtype=np.int8)
        
        np.testing.assert_array_equal(grid, expected_grid)
        
        # Test with invert=True
        inverted_grid = depth_to_grid(
            depth_map=self.depth_map,
            grid_size=4,
            threshold_factor=0.5,
            smoothing=False,
            invert=True
        )
        
        # Check inverted thresholding (values > 0.5 should be obstacles)
        expected_inverted = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ], dtype=np.int8)
        
        np.testing.assert_array_equal(inverted_grid, expected_inverted)
    
    def test_add_start_goal_points(self) -> None:
        """Test the add_start_goal_points function."""
        # Create a simple grid
        grid = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ], dtype=np.int8)
        
        # Add start and goal points
        start_pos = (0, 0)
        goal_pos = (3, 3)
        
        grid_with_points = add_start_goal_points(grid, start_pos, goal_pos)
        
        # Check that start and goal positions are set correctly
        self.assertEqual(grid_with_points[start_pos], START_POINT)
        self.assertEqual(grid_with_points[goal_pos], GOAL_POINT)
        
        # Check that the original grid was not modified
        self.assertEqual(grid[start_pos], FREE_SPACE)
        self.assertEqual(grid[goal_pos], FREE_SPACE)
        
        # Test with default goal position
        grid_with_default_goal = add_start_goal_points(grid, start_pos)
        self.assertEqual(grid_with_default_goal[start_pos], START_POINT)
        self.assertEqual(grid_with_default_goal[3, 3], GOAL_POINT)
        
        # Test with invalid positions
        with self.assertRaises(ValueError):
            add_start_goal_points(grid, (-1, 0))
        
        with self.assertRaises(ValueError):
            add_start_goal_points(grid, (0, 0), (4, 4))
    
    def test_grid_size_resizing(self) -> None:
        """Test that the grid is properly resized."""
        # Convert to a different grid size
        grid_size = 8
        grid = depth_to_grid(
            depth_map=self.depth_map,
            grid_size=grid_size,
            threshold_factor=0.5,
            smoothing=False
        )
        
        # Check output shape
        self.assertEqual(grid.shape, (grid_size, grid_size))
    
    def test_invalid_parameters(self) -> None:
        """Test handling of invalid parameters."""
        # Test invalid grid size
        with self.assertRaises(ValueError):
            depth_to_grid(
                depth_map=self.depth_map,
                grid_size=-1,
                threshold_factor=0.5,
                smoothing=False
            )
        
        # Test invalid threshold
        with self.assertRaises(ValueError):
            depth_to_grid(
                depth_map=self.depth_map,
                grid_size=4,
                threshold_factor=1.5,  # Should be in range [0, 1]
                smoothing=False
            )


if __name__ == "__main__":
    unittest.main() 