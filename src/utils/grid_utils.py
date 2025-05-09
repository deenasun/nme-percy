"""Utility functions for grid operations."""

from typing import Tuple, List, Optional, Union, Any
import numpy as np


def find_start_goal_positions(grid: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Find optimal start and goal positions in a grid.
    
    Start position is the topmost-leftmost non-obstacle cell.
    Goal position is the bottommost-rightmost non-obstacle cell.
    
    Args:
        grid: 2D numpy array where 0 represents free space and 1 represents obstacles
        
    Returns:
        Tuple of (start_position, goal_position) where each position is (row, col)
        
    Raises:
        ValueError: If no valid start or goal positions can be found
    """
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("Grid must be a 2D numpy array")
    
    rows, cols = grid.shape
    
    # Find start position (topmost-leftmost free cell)
    start_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:  # Free space
                start_pos = (r, c)
                break
        if start_pos is not None:
            break
    
    # Find goal position (bottommost-rightmost free cell)
    goal_pos = None
    for r in range(rows-1, -1, -1):
        for c in range(cols-1, -1, -1):
            if grid[r, c] == 0:  # Free space
                goal_pos = (r, c)
                break
        if goal_pos is not None:
            break
    
    # Verify that we found valid positions
    if start_pos is None:
        raise ValueError("Could not find a valid start position (no free cells)")
    if goal_pos is None:
        raise ValueError("Could not find a valid goal position (no free cells)")
    
    # If start and goal are the same, try to find an alternative goal
    if start_pos == goal_pos:
        # Try to find any other free cell
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0 and (r, c) != start_pos:
                    goal_pos = (r, c)
                    break
            if start_pos != goal_pos:
                break
    
    # Final verification
    if start_pos == goal_pos:
        raise ValueError("Grid has only one free cell - cannot have distinct start and goal positions")
    
    return start_pos, goal_pos 