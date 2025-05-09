"""A* search implementation for grid-based path planning.

This module implements the A* search algorithm for finding optimal paths
through grid environments where 0 represents free space and 1 represents obstacles.
"""

from typing import List, Tuple, Dict, Set, Optional, Callable, Union, Any
import heapq
import numpy as np
import time


Position = Tuple[int, int]
Path = List[Position]
Grid = np.ndarray


def manhattan_distance(a: Position, b: Position) -> int:
    """Calculate the Manhattan distance between two points.
    
    Manhattan distance is the sum of the absolute differences of 
    their Cartesian coordinates.
    
    Args:
        a: First position (row, col)
        b: Second position (row, col)
        
    Returns:
        The Manhattan distance between positions a and b
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a: Position, b: Position) -> float:
    """Calculate the Euclidean distance between two points.
    
    Euclidean distance is the straight-line distance between two points.
    
    Args:
        a: First position (row, col)
        b: Second position (row, col)
        
    Returns:
        The Euclidean distance between positions a and b
    """
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5


def chebyshev_distance(a: Position, b: Position) -> int:
    """Calculate the Chebyshev distance between two points.
    
    Chebyshev distance is the maximum of the absolute differences of 
    their Cartesian coordinates.
    
    Args:
        a: First position (row, col)
        b: Second position (row, col)
        
    Returns:
        The Chebyshev distance between positions a and b
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def get_neighbors(
    position: Position, 
    grid: Grid, 
    diagonal: bool = False
) -> List[Position]:
    """Get valid neighboring positions in the grid.
    
    Args:
        position: Current position (row, col)
        grid: Grid environment (0 for free space, 1 for obstacle)
        diagonal: Whether to include diagonal neighbors
        
    Returns:
        List of valid neighboring positions
    """
    row, col = position
    if diagonal:
        # Include 8 neighbors (including diagonals)
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1), 
            (0, -1),           (0, 1), 
            (1, -1),  (1, 0),  (1, 1)
        ]
    else:
        # Include only 4 neighbors (up, right, down, left)
        neighbor_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    neighbors = []
    for dr, dc in neighbor_offsets:
        new_row, new_col = row + dr, col + dc
        
        # Check grid boundaries
        if (new_row < 0 or new_row >= grid.shape[0] or 
            new_col < 0 or new_col >= grid.shape[1]):
            continue
        
        # Check if obstacle
        if grid[new_row, new_col] == 1:
            continue
        
        neighbors.append((new_row, new_col))
    
    return neighbors


def reconstruct_path(
    came_from: Dict[Position, Position], 
    start: Position, 
    goal: Position
) -> Path:
    """Reconstruct the path from start to goal using the came_from dictionary.
    
    Args:
        came_from: Dictionary mapping each position to the position it came from
        start: Starting position
        goal: Goal position
        
    Returns:
        List of positions from start to goal
    """
    path = []
    current = goal
    
    if goal not in came_from and start != goal:
        return []  # No path exists
    
    while current != start:
        path.append(current)
        if current not in came_from:
            return []  # This should not happen if the path exists
        current = came_from[current]
    
    path.append(start)
    return path[::-1]  # Return reversed path


def a_star_search(
    grid: Grid, 
    start: Position, 
    goal: Position, 
    heuristic: Union[str, Callable[[Position, Position], float]] = "manhattan",
    diagonal: bool = False,
    timeout: Optional[float] = None,
    weight: float = 1.0
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Find the optimal path through a grid using A* search algorithm.
    
    Args:
        grid: Grid environment (0 for free space, 1 for obstacle)
        start: Starting position (row, col)
        goal: Goal position (row, col)
        heuristic: Heuristic function to use:
            - "manhattan": Manhattan distance (default)
            - "euclidean": Euclidean distance
            - "chebyshev": Chebyshev distance
            - Custom function: Takes two positions and returns a float
        diagonal: Whether to allow diagonal movements
        timeout: Maximum time in seconds to run the algorithm (None for no limit)
        weight: Weight for the heuristic (1.0 for regular A*)
    
    Returns:
        Tuple containing:
            - The path from start to goal (or None if no path exists)
            - Dictionary with algorithm metrics:
                - nodes_explored: Number of nodes explored
                - path_length: Length of the found path
                - time_taken: Time taken to find the path
                - path_cost: Total cost of the path
    """
    # Check if start or goal is an obstacle
    if grid[start] == 1 or grid[goal] == 1:
        return None, {"nodes_explored": 0, "path_length": 0, "time_taken": 0, "path_cost": float('inf')}
    
    # If start is the goal, return a path with just the goal
    if start == goal:
        return [start], {"nodes_explored": 0, "path_length": 1, "time_taken": 0, "path_cost": 0}
    
    # Select the heuristic function
    if heuristic == "manhattan":
        h_func = manhattan_distance
    elif heuristic == "euclidean":
        h_func = euclidean_distance
    elif heuristic == "chebyshev":
        h_func = chebyshev_distance
    elif callable(heuristic):
        h_func = heuristic
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
    
    # Start the timer
    start_time = time.time()
    
    # Initialize data structures
    closed_set: Set[Position] = set()
    came_from: Dict[Position, Position] = {}
    g_score: Dict[Position, float] = {start: 0}
    f_score: Dict[Position, float] = {start: weight * h_func(start, goal)}
    open_set = []
    heapq.heappush(open_set, (f_score[start], start))
    
    # Track metrics
    nodes_explored = 0
    
    # Main loop
    while open_set:
        # Check timeout
        if timeout and time.time() - start_time > timeout:
            # Return the best path found so far
            current = min((f_score.get(pos, float('inf')), pos) for pos in g_score.keys() if pos not in closed_set)[1]
            path = reconstruct_path(came_from, start, current)
            return path, {
                "nodes_explored": nodes_explored,
                "path_length": len(path),
                "time_taken": time.time() - start_time,
                "path_cost": g_score.get(current, float('inf')),
                "timeout": True
            }
        
        # Get the position with the lowest f_score
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        
        # Check if we've reached the goal
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, {
                "nodes_explored": nodes_explored,
                "path_length": len(path),
                "time_taken": time.time() - start_time,
                "path_cost": g_score[goal]
            }
        
        # Mark current position as visited
        closed_set.add(current)
        
        # Check all valid neighbors
        for neighbor in get_neighbors(current, grid, diagonal):
            # Skip if already evaluated
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g_score (cost to reach neighbor through current)
            movement_cost = 1.0
            if diagonal and abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) > 1:
                movement_cost = 1.414  # sqrt(2) for diagonal movement
            
            tentative_g_score = g_score[current] + movement_cost
            
            # Skip if we already have a better path to neighbor
            if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                continue
            
            # This is the best path to neighbor so far
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + weight * h_func(neighbor, goal)
            
            # Add neighbor to open set if not already there
            if neighbor not in [i[1] for i in open_set]:
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None, {
        "nodes_explored": nodes_explored,
        "path_length": 0,
        "time_taken": time.time() - start_time,
        "path_cost": float('inf')
    }


def bidirectional_a_star(
    grid: Grid, 
    start: Position, 
    goal: Position, 
    heuristic: Union[str, Callable[[Position, Position], float]] = "manhattan",
    diagonal: bool = False,
    timeout: Optional[float] = None
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Find the optimal path using bidirectional A* search.
    
    Searches from both start and goal simultaneously.
    
    Args:
        grid: Grid environment (0 for free space, 1 for obstacle)
        start: Starting position (row, col)
        goal: Goal position (row, col)
        heuristic: Heuristic function to use
        diagonal: Whether to allow diagonal movements
        timeout: Maximum time in seconds to run the algorithm
    
    Returns:
        Tuple containing the path and metrics dictionary
    """
    # Select the heuristic function
    if heuristic == "manhattan":
        h_func = manhattan_distance
    elif heuristic == "euclidean":
        h_func = euclidean_distance
    elif heuristic == "chebyshev":
        h_func = chebyshev_distance
    elif callable(heuristic):
        h_func = heuristic
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
    
    # Start the timer
    start_time = time.time()
    
    # Initialize forward search
    forward_closed: Set[Position] = set()
    forward_came_from: Dict[Position, Position] = {}
    forward_g_score: Dict[Position, float] = {start: 0}
    forward_f_score: Dict[Position, float] = {start: h_func(start, goal)}
    forward_open = []
    heapq.heappush(forward_open, (forward_f_score[start], start))
    
    # Initialize backward search
    backward_closed: Set[Position] = set()
    backward_came_from: Dict[Position, Position] = {}
    backward_g_score: Dict[Position, float] = {goal: 0}
    backward_f_score: Dict[Position, float] = {goal: h_func(goal, start)}
    backward_open = []
    heapq.heappush(backward_open, (backward_f_score[goal], goal))
    
    # Track metrics
    nodes_explored = 0
    
    # Track the best meeting point and path length
    best_meeting_point = None
    best_path_length = float('inf')
    
    # Main loop
    while forward_open and backward_open:
        # Check timeout
        if timeout and time.time() - start_time > timeout:
            # Construct the best path found so far
            if best_meeting_point:
                forward_path = reconstruct_path(forward_came_from, start, best_meeting_point)
                backward_path = reconstruct_path(backward_came_from, goal, best_meeting_point)
                backward_path.pop(0)  # Remove duplicate meeting point
                path = forward_path + backward_path[::-1]
                return path, {
                    "nodes_explored": nodes_explored,
                    "path_length": len(path),
                    "time_taken": time.time() - start_time,
                    "path_cost": forward_g_score.get(best_meeting_point, 0) + backward_g_score.get(best_meeting_point, 0),
                    "timeout": True
                }
            return None, {
                "nodes_explored": nodes_explored,
                "path_length": 0,
                "time_taken": time.time() - start_time,
                "path_cost": float('inf'),
                "timeout": True
            }
        
        # Forward search step
        _, current_forward = heapq.heappop(forward_open)
        nodes_explored += 1
        
        # Check if current_forward has been visited by backward search
        if current_forward in backward_closed:
            # Path found, calculate total path length
            path_length = forward_g_score[current_forward] + backward_g_score[current_forward]
            if path_length < best_path_length:
                best_meeting_point = current_forward
                best_path_length = path_length
        
        forward_closed.add(current_forward)
        
        # Explore neighbors in forward search
        for neighbor in get_neighbors(current_forward, grid, diagonal):
            if neighbor in forward_closed:
                continue
                
            # Calculate movement cost
            movement_cost = 1.0
            if diagonal and abs(neighbor[0] - current_forward[0]) + abs(neighbor[1] - current_forward[1]) > 1:
                movement_cost = 1.414  # sqrt(2) for diagonal movement
                
            tentative_g_score = forward_g_score[current_forward] + movement_cost
            
            if (neighbor not in forward_g_score or 
                tentative_g_score < forward_g_score[neighbor]):
                forward_came_from[neighbor] = current_forward
                forward_g_score[neighbor] = tentative_g_score
                forward_f_score[neighbor] = tentative_g_score + h_func(neighbor, goal)
                if neighbor not in [i[1] for i in forward_open]:
                    heapq.heappush(forward_open, (forward_f_score[neighbor], neighbor))
        
        # Backward search step
        _, current_backward = heapq.heappop(backward_open)
        nodes_explored += 1
        
        # Check if current_backward has been visited by forward search
        if current_backward in forward_closed:
            # Path found, calculate total path length
            path_length = forward_g_score[current_backward] + backward_g_score[current_backward]
            if path_length < best_path_length:
                best_meeting_point = current_backward
                best_path_length = path_length
        
        backward_closed.add(current_backward)
        
        # Explore neighbors in backward search
        for neighbor in get_neighbors(current_backward, grid, diagonal):
            if neighbor in backward_closed:
                continue
                
            # Calculate movement cost
            movement_cost = 1.0
            if diagonal and abs(neighbor[0] - current_backward[0]) + abs(neighbor[1] - current_backward[1]) > 1:
                movement_cost = 1.414  # sqrt(2) for diagonal movement
                
            tentative_g_score = backward_g_score[current_backward] + movement_cost
            
            if (neighbor not in backward_g_score or 
                tentative_g_score < backward_g_score[neighbor]):
                backward_came_from[neighbor] = current_backward
                backward_g_score[neighbor] = tentative_g_score
                backward_f_score[neighbor] = tentative_g_score + h_func(neighbor, start)
                if neighbor not in [i[1] for i in backward_open]:
                    heapq.heappush(backward_open, (backward_f_score[neighbor], neighbor))
        
        # Check if we've found the best path
        if best_meeting_point and len(forward_open) > 0 and len(backward_open) > 0:
            # Get the minimum f-scores from both open sets
            min_forward_f = forward_open[0][0]
            min_backward_f = backward_open[0][0]
            
            # If the sum of minimum f-scores is >= best path length, we've found the optimal path
            if min_forward_f + min_backward_f >= best_path_length:
                # Construct the path
                forward_path = reconstruct_path(forward_came_from, start, best_meeting_point)
                backward_path = reconstruct_path(backward_came_from, goal, best_meeting_point)
                backward_path.pop(0)  # Remove duplicate meeting point
                path = forward_path + backward_path[::-1]
                return path, {
                    "nodes_explored": nodes_explored,
                    "path_length": len(path),
                    "time_taken": time.time() - start_time,
                    "path_cost": best_path_length
                }
    
    # If we've found a meeting point but didn't have a chance to construct the path
    if best_meeting_point:
        forward_path = reconstruct_path(forward_came_from, start, best_meeting_point)
        backward_path = reconstruct_path(backward_came_from, goal, best_meeting_point)
        backward_path.pop(0)  # Remove duplicate meeting point
        path = forward_path + backward_path[::-1]
        return path, {
            "nodes_explored": nodes_explored,
            "path_length": len(path),
            "time_taken": time.time() - start_time,
            "path_cost": best_path_length
        }
    
    # No path found
    return None, {
        "nodes_explored": nodes_explored,
        "path_length": 0,
        "time_taken": time.time() - start_time,
        "path_cost": float('inf')
    } 