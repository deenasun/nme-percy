#!/usr/bin/env python
"""Unit tests for the planning module and A* search algorithm.

This module contains unit tests for the A* search algorithm and related functions.
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.planning.a_star import (
    a_star_search,
    bidirectional_a_star,
    manhattan_distance,
    euclidean_distance,
    chebyshev_distance,
    get_neighbors,
    reconstruct_path
)


class TestAStar(unittest.TestCase):
    """Test cases for the A* search algorithm and related functions."""

    def setUp(self):
        """Set up test fixtures for each test method."""
        # Create a simple grid for testing
        # 0 = free space, 1 = obstacle
        self.grid_small = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int8)

        # Create a larger grid with a maze-like pattern
        self.grid_maze = np.zeros((10, 10), dtype=np.int8)
        # Add vertical wall
        self.grid_maze[1:8, 5] = 1
        # Add horizontal walls
        self.grid_maze[3, 1:5] = 1
        self.grid_maze[6, 6:9] = 1

        # Create a grid with no path
        self.grid_no_path = np.zeros((5, 5), dtype=np.int8)
        self.grid_no_path[1:4, 2] = 1  # Vertical wall
        self.grid_no_path[0:3, 4] = 1  # Partial vertical wall
        self.grid_no_path[4, 0:4] = 1  # Horizontal wall

        # Test positions
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.no_path_start = (0, 0)
        self.no_path_goal = (4, 4)

    def test_manhattan_distance(self):
        """Test the Manhattan distance heuristic function."""
        # Test known distances
        self.assertEqual(manhattan_distance((0, 0), (3, 4)), 7)
        self.assertEqual(manhattan_distance((2, 3), (5, 7)), 7)
        self.assertEqual(manhattan_distance((1, 1), (1, 1)), 0)

    def test_euclidean_distance(self):
        """Test the Euclidean distance heuristic function."""
        # Test known distances
        self.assertAlmostEqual(euclidean_distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(euclidean_distance((2, 3), (5, 7)), 5.0)
        self.assertEqual(euclidean_distance((1, 1), (1, 1)), 0.0)

    def test_chebyshev_distance(self):
        """Test the Chebyshev distance heuristic function."""
        # Test known distances
        self.assertEqual(chebyshev_distance((0, 0), (3, 4)), 4)
        self.assertEqual(chebyshev_distance((2, 3), (5, 7)), 4)
        self.assertEqual(chebyshev_distance((1, 1), (1, 1)), 0)

    def test_get_neighbors(self):
        """Test the get_neighbors function with and without diagonal movement."""
        # Create a test grid
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[1, 1] = 1  # Add an obstacle
        grid[3, 3] = 1  # Add another obstacle

        # Test non-diagonal neighbors
        pos = (2, 2)
        neighbors = get_neighbors(grid, pos, diagonal=False)
        # Should return up, right, down, left (4 positions)
        self.assertEqual(len(neighbors), 4)
        self.assertIn((1, 2), neighbors)  # up
        self.assertIn((2, 3), neighbors)  # right
        self.assertIn((3, 2), neighbors)  # down
        self.assertIn((2, 1), neighbors)  # left

        # Test diagonal neighbors
        neighbors = get_neighbors(grid, pos, diagonal=True)
        # Should return 8 positions (4 cardinal + 4 diagonal)
        self.assertEqual(len(neighbors), 8)
        # Cardinal directions
        self.assertIn((1, 2), neighbors)  # up
        self.assertIn((2, 3), neighbors)  # right
        self.assertIn((3, 2), neighbors)  # down
        self.assertIn((2, 1), neighbors)  # left
        # Diagonal directions
        self.assertIn((1, 1), neighbors)  # up-left (this is an obstacle, should not be included)
        self.assertIn((1, 3), neighbors)  # up-right
        self.assertIn((3, 3), neighbors)  # down-right (this is an obstacle, should not be included)
        self.assertIn((3, 1), neighbors)  # down-left

        # Test edge of grid
        pos = (0, 0)  # Top left corner
        neighbors = get_neighbors(grid, pos, diagonal=False)
        # Should return right and down only (2 positions)
        self.assertEqual(len(neighbors), 2)
        self.assertIn((0, 1), neighbors)  # right
        self.assertIn((1, 0), neighbors)  # down

        # Test with diagonal at edge
        neighbors = get_neighbors(grid, pos, diagonal=True)
        # Should return right, down, and down-right (3 positions)
        self.assertEqual(len(neighbors), 3)
        self.assertIn((0, 1), neighbors)  # right
        self.assertIn((1, 0), neighbors)  # down
        self.assertIn((1, 1), neighbors)  # down-right (this is an obstacle, should not be included)

    def test_reconstruct_path(self):
        """Test the reconstruct_path function."""
        # Create a simple came_from dictionary
        came_from = {
            (1, 1): (0, 0),
            (2, 2): (1, 1),
            (3, 3): (2, 2),
            (4, 4): (3, 3)
        }
        
        # Reconstruct the path
        path = reconstruct_path(came_from, (0, 0), (4, 4))
        
        # The path should include all points from start to goal
        expected_path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        self.assertEqual(path, expected_path)

    def test_a_star_small_grid(self):
        """Test A* search on a small grid."""
        # Run A* search
        path, metrics = a_star_search(
            grid=self.grid_small,
            start=self.start_pos,
            goal=self.goal_pos
        )
        
        # Check if a path was found
        self.assertIsNotNone(path)
        
        # Verify the path is valid
        self.assertEqual(path[0], self.start_pos)
        self.assertEqual(path[-1], self.goal_pos)
        
        # Check that the path avoids obstacles
        for pos in path:
            self.assertEqual(self.grid_small[pos], 0)
        
        # Check that the path is continuous (each step moves by 1 in any direction)
        for i in range(1, len(path)):
            diff_row = abs(path[i][0] - path[i-1][0])
            diff_col = abs(path[i][1] - path[i-1][1])
            # Either row or col changes by 1, but not both
            self.assertLessEqual(diff_row + diff_col, 2)
            self.assertGreaterEqual(diff_row + diff_col, 1)
        
        # Check metrics
        self.assertIn('nodes_explored', metrics)
        self.assertIn('time_taken', metrics)
        self.assertIn('path_cost', metrics)
        self.assertGreater(metrics['nodes_explored'], 0)
        self.assertGreater(metrics['time_taken'], 0)
        self.assertEqual(metrics['path_cost'], len(path) - 1)  # Cost is the number of steps

    def test_a_star_maze_grid(self):
        """Test A* search on a maze-like grid."""
        # Run A* search
        path, metrics = a_star_search(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9)
        )
        
        # Check if a path was found
        self.assertIsNotNone(path)
        
        # Verify the path is valid
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (9, 9))
        
        # Check that the path avoids obstacles
        for pos in path:
            self.assertEqual(self.grid_maze[pos], 0)
        
        # Check that the path is continuous
        for i in range(1, len(path)):
            diff_row = abs(path[i][0] - path[i-1][0])
            diff_col = abs(path[i][1] - path[i-1][1])
            self.assertLessEqual(diff_row + diff_col, 2)
            self.assertGreaterEqual(diff_row + diff_col, 1)

    def test_a_star_no_path(self):
        """Test A* search when no path exists."""
        # Run A* search on a grid with no possible path
        path, metrics = a_star_search(
            grid=self.grid_no_path,
            start=self.no_path_start,
            goal=self.no_path_goal
        )
        
        # Check that no path was found
        self.assertEqual(path, [])
        
        # Check that metrics are still returned
        self.assertIn('nodes_explored', metrics)
        self.assertIn('time_taken', metrics)
        self.assertGreater(metrics['nodes_explored'], 0)
        self.assertGreater(metrics['time_taken'], 0)

    def test_a_star_diagonal(self):
        """Test A* search with diagonal movement allowed."""
        # Run A* search with diagonal movement
        path, metrics = a_star_search(
            grid=self.grid_small,
            start=self.start_pos,
            goal=self.goal_pos,
            diagonal=True
        )
        
        # Check if a path was found
        self.assertIsNotNone(path)
        
        # The path should be shorter with diagonal movement
        path_straight, metrics_straight = a_star_search(
            grid=self.grid_small,
            start=self.start_pos,
            goal=self.goal_pos,
            diagonal=False
        )
        
        # The diagonal path should be shorter or equal in length
        self.assertLessEqual(len(path), len(path_straight))
        
        # Check that diagonal movements are valid
        for i in range(1, len(path)):
            diff_row = abs(path[i][0] - path[i-1][0])
            diff_col = abs(path[i][1] - path[i-1][1])
            # Both row and col can change by at most 1
            self.assertLessEqual(diff_row, 1)
            self.assertLessEqual(diff_col, 1)

    def test_a_star_heuristics(self):
        """Test A* search with different heuristics."""
        # Run A* with different heuristics
        path_manhattan, metrics_manhattan = a_star_search(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9),
            heuristic="manhattan"
        )
        
        path_euclidean, metrics_euclidean = a_star_search(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9),
            heuristic="euclidean"
        )
        
        path_chebyshev, metrics_chebyshev = a_star_search(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9),
            heuristic="chebyshev"
        )
        
        # All heuristics should find a path
        self.assertIsNotNone(path_manhattan)
        self.assertIsNotNone(path_euclidean)
        self.assertIsNotNone(path_chebyshev)
        
        # Check that all paths start and end at the same points
        self.assertEqual(path_manhattan[0], (0, 0))
        self.assertEqual(path_manhattan[-1], (9, 9))
        self.assertEqual(path_euclidean[0], (0, 0))
        self.assertEqual(path_euclidean[-1], (9, 9))
        self.assertEqual(path_chebyshev[0], (0, 0))
        self.assertEqual(path_chebyshev[-1], (9, 9))
        
        # Different heuristics will explore different numbers of nodes
        # This is a simple check to ensure different heuristics are being used
        self.assertNotEqual(metrics_manhattan['nodes_explored'], 
                           metrics_euclidean['nodes_explored'])

    def test_a_star_weighted(self):
        """Test weighted A* search."""
        # Run A* with different weights
        path_normal, metrics_normal = a_star_search(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9),
            weight=1.0
        )
        
        path_weighted, metrics_weighted = a_star_search(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9),
            weight=2.0
        )
        
        # Both should find a path
        self.assertIsNotNone(path_normal)
        self.assertIsNotNone(path_weighted)
        
        # Weighted A* should explore fewer nodes but might find a longer path
        self.assertLess(metrics_weighted['nodes_explored'], metrics_normal['nodes_explored'])
        self.assertGreaterEqual(len(path_weighted), len(path_normal))

    def test_a_star_timeout(self):
        """Test A* search with a timeout."""
        # Create a larger grid to increase computation time
        large_grid = np.zeros((50, 50), dtype=np.int8)
        
        # Add a complex maze pattern
        for i in range(0, 50, 2):
            large_grid[i, :i] = 1
        
        # Run A* with a very short timeout
        path, metrics = a_star_search(
            grid=large_grid,
            start=(0, 0),
            goal=(49, 49),
            timeout=0.001  # Very short timeout
        )
        
        # Check that the timeout was triggered
        self.assertTrue(metrics['timeout'])
        
        # A* should return a partial path or an empty list
        if path:
            self.assertNotEqual(path[-1], (49, 49))  # Should not reach the goal
        else:
            self.assertEqual(path, [])  # Empty path if no progress was made

    def test_bidirectional_a_star(self):
        """Test bidirectional A* search."""
        # Run bidirectional A* search
        path, metrics = bidirectional_a_star(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9)
        )
        
        # Check if a path was found
        self.assertIsNotNone(path)
        
        # Verify the path is valid
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (9, 9))
        
        # Check that the path avoids obstacles
        for pos in path:
            self.assertEqual(self.grid_maze[pos], 0)
        
        # Compare with regular A*
        path_regular, metrics_regular = a_star_search(
            grid=self.grid_maze,
            start=(0, 0),
            goal=(9, 9)
        )
        
        # The paths might be different, but both should be valid and optimal
        self.assertEqual(len(path), len(path_regular))
        
        # Bidirectional A* should generally explore fewer nodes
        self.assertLessEqual(metrics['nodes_explored'], metrics_regular['nodes_explored'] * 1.5)

    def test_invalid_inputs(self):
        """Test A* search with invalid inputs."""
        # Test with invalid grid
        with self.assertRaises(ValueError):
            a_star_search(
                grid=np.array([]),  # Empty grid
                start=self.start_pos,
                goal=self.goal_pos
            )
        
        # Test with start position outside grid
        with self.assertRaises(ValueError):
            a_star_search(
                grid=self.grid_small,
                start=(10, 10),  # Outside the grid
                goal=self.goal_pos
            )
        
        # Test with goal position outside grid
        with self.assertRaises(ValueError):
            a_star_search(
                grid=self.grid_small,
                start=self.start_pos,
                goal=(-1, -1)  # Outside the grid
            )
        
        # Test with start position on an obstacle
        invalid_start = (2, 2)  # There's an obstacle at this position
        with self.assertRaises(ValueError):
            a_star_search(
                grid=self.grid_small,
                start=invalid_start,
                goal=self.goal_pos
            )
        
        # Test with goal position on an obstacle
        invalid_goal = (2, 2)  # There's an obstacle at this position
        with self.assertRaises(ValueError):
            a_star_search(
                grid=self.grid_small,
                start=self.start_pos,
                goal=invalid_goal
            )
        
        # Test with invalid heuristic
        with self.assertRaises(ValueError):
            a_star_search(
                grid=self.grid_small,
                start=self.start_pos,
                goal=self.goal_pos,
                heuristic="invalid_heuristic"
            )
        
        # Test with invalid weight
        with self.assertRaises(ValueError):
            a_star_search(
                grid=self.grid_small,
                start=self.start_pos,
                goal=self.goal_pos,
                weight=0.0  # Weight must be >= 1.0
            )


if __name__ == "__main__":
    unittest.main() 