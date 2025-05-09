"""Utility functions for testing and debugging the navigation system."""

import numpy as np
from typing import Tuple, Optional

def create_test_grid(
    size: int = 8, 
    start_pos: Tuple[int, int] = (0, 0),
    goal_pos: Optional[Tuple[int, int]] = None,
    pattern: str = "empty"
) -> np.ndarray:
    """Create a simple test grid for debugging the navigation system.
    
    Args:
        size: Size of the grid (size x size)
        start_pos: Starting position (row, col)
        goal_pos: Goal position (row, col), if None, defaults to (size-1, size-1)
        pattern: Grid pattern type, options:
            - "empty": No obstacles, completely open grid
            - "border": Obstacles around the border only
            - "corridor": A corridor from start to goal
            - "maze": A simple maze pattern with a guaranteed path
            
    Returns:
        A numpy array representing the grid (0 = free, 1 = obstacle)
    """
    # Set default goal position
    if goal_pos is None:
        goal_pos = (size-1, size-1)
    
    # Create base grid (all free space)
    grid = np.zeros((size, size), dtype=np.int8)
    
    if pattern == "empty":
        # Empty grid, all zeros (no obstacles)
        pass
    
    elif pattern == "border":
        # Add obstacles around the border
        grid[0, :] = 1  # Top border
        grid[size-1, :] = 1  # Bottom border
        grid[:, 0] = 1  # Left border
        grid[:, size-1] = 1  # Right border
        
        # Ensure start and goal positions are free
        grid[start_pos] = 0
        grid[goal_pos] = 0
    
    elif pattern == "corridor":
        # Fill with obstacles
        grid.fill(1)
        
        # Create a corridor from start to goal
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos
        
        # Horizontal path to the goal column
        for col in range(min(start_col, goal_col), max(start_col, goal_col) + 1):
            grid[start_row, col] = 0
        
        # Vertical path to the goal row
        for row in range(min(start_row, goal_row), max(start_row, goal_row) + 1):
            grid[row, goal_col] = 0
    
    elif pattern == "maze":
        # Create a simple maze with guaranteed path
        # Fill with obstacles
        grid.fill(1)
        
        # Create a maze-like pattern with a guaranteed path from start to goal
        # First, create a path from start to goal
        current_pos = start_pos
        target_pos = goal_pos
        
        # Mark starting position as free
        grid[current_pos] = 0
        
        # We'll use a simple approach:
        # 1. Move horizontally until we reach the goal column
        # 2. Move vertically until we reach the goal row
        
        # Horizontal movement
        current_row, current_col = current_pos
        target_row, target_col = target_pos
        
        # Add some randomness to the pattern
        np.random.seed(42)  # For reproducibility
        
        # First, create a random path from start to goal
        path = []
        pos = start_pos
        while pos != goal_pos:
            path.append(pos)
            row, col = pos
            
            # Determine possible moves (up, right, down, left)
            # that would bring us closer to the goal
            possible_moves = []
            
            if row > 0 and row > target_row:
                possible_moves.append((-1, 0))  # Up
            if col < size-1 and col < target_col:
                possible_moves.append((0, 1))  # Right
            if row < size-1 and row < target_row:
                possible_moves.append((1, 0))  # Down
            if col > 0 and col > target_col:
                possible_moves.append((0, -1))  # Left
            
            # If no progress moves, just move in any valid direction
            if not possible_moves:
                if row > 0:
                    possible_moves.append((-1, 0))  # Up
                if col < size-1:
                    possible_moves.append((0, 1))  # Right
                if row < size-1:
                    possible_moves.append((1, 0))  # Down
                if col > 0:
                    possible_moves.append((0, -1))  # Left
            
            # Choose a random move
            if possible_moves:
                dr, dc = possible_moves[np.random.choice(len(possible_moves))]
                pos = (row + dr, col + dc)
        
        # Add goal to path
        path.append(goal_pos)
        
        # Mark the path as free space
        for pos in path:
            grid[pos] = 0
        
        # Add some random free spaces (30% of remaining cells)
        obstacles = np.where(grid == 1)
        num_to_free = int(len(obstacles[0]) * 0.3)
        indices = np.random.choice(len(obstacles[0]), num_to_free, replace=False)
        
        for idx in indices:
            row, col = obstacles[0][idx], obstacles[1][idx]
            grid[row, col] = 0
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Ensure start and goal positions are free
    grid[start_pos] = 0
    grid[goal_pos] = 0
    
    return grid

def diagnose_q_table(q_table, grid_size, start_pos, goal_pos):
    """Diagnose issues with the Q-table.
    
    Args:
        q_table: The Q-table to diagnose
        grid_size: Size of the grid
        start_pos: Starting position (row, col)
        goal_pos: Goal position (row, col)
        
    Returns:
        A diagnostic report as a string
    """
    # Get Q-values statistics
    min_q = q_table.min()
    max_q = q_table.max()
    mean_q = q_table.mean()
    non_zero = (q_table != 0).sum()
    total_cells = q_table.size
    
    # Check Q-values from start to goal
    path = []
    current_pos = start_pos
    max_steps = grid_size * 2  # Avoid infinite loops
    steps = 0
    
    while current_pos != goal_pos and steps < max_steps:
        row, col = current_pos
        goal_row, goal_col = goal_pos
        
        # Get best action at current state
        action = np.argmax(q_table[row, col, goal_row, goal_col])
        
        # Add to path
        path.append((current_pos, action))
        
        # Move according to action
        if action == 0:  # Up
            next_pos = (max(0, row - 1), col)
        elif action == 1:  # Right
            next_pos = (row, min(grid_size - 1, col + 1))
        elif action == 2:  # Down
            next_pos = (min(grid_size - 1, row + 1), col)
        elif action == 3:  # Left
            next_pos = (row, max(0, col - 1))
        else:
            next_pos = current_pos  # Invalid action
        
        # Update position
        current_pos = next_pos
        steps += 1
    
    # Build diagnostic report
    report = "Q-Table Diagnostic Report\n"
    report += "======================\n"
    report += f"Grid Size: {grid_size}x{grid_size}\n"
    report += f"Start Position: {start_pos}\n"
    report += f"Goal Position: {goal_pos}\n"
    report += f"Q-values: min={min_q}, max={max_q}, mean={mean_q}\n"
    report += f"Non-zero values: {non_zero}/{total_cells} ({non_zero/total_cells:.2%})\n"
    
    if current_pos == goal_pos:
        report += f"Path found from start to goal in {steps} steps\n"
    else:
        report += f"Failed to find path from start to goal (stopped after {steps} steps)\n"
    
    report += "\nPath from start:\n"
    for i, (pos, action) in enumerate(path):
        action_name = ["Up", "Right", "Down", "Left"][action]
        report += f"{i+1}. Position {pos}, Action: {action_name}\n"
    
    return report 