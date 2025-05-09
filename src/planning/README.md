# Path Planning Algorithms

This module contains algorithms for finding optimal paths through grid environments.

## A* Search Algorithm

A* (pronounced "A-star") is a best-first search algorithm that finds the shortest path from a start node to a goal node. It combines the advantages of both Dijkstra's algorithm (guaranteeing the shortest path) and greedy best-first search (using heuristics to speed up the search).

### Features

- **Optimal**: Guaranteed to find the shortest path if a valid path exists.
- **Efficient**: Uses heuristics to guide the search toward the goal.
- **Flexible**: Supports different heuristics and diagonal movements.
- **Configurable**: Adjust parameters like timeout and weight to balance speed and optimality.
- **Informative**: Returns detailed metrics about the search process.

### Heuristics

The A* algorithm supports multiple heuristics to estimate the distance from a node to the goal:

1. **Manhattan Distance (default)**: Sum of the absolute differences of the coordinates. Optimal for 4-connected grids (no diagonal movement).
2. **Euclidean Distance**: Straight-line distance between points. Good for continuous spaces or when diagonal movement costs sqrt(2).
3. **Chebyshev Distance**: Maximum of the absolute differences of the coordinates. Optimal for 8-connected grids where diagonal movement costs 1.
4. **Custom Heuristic**: You can provide your own heuristic function.

### Basic Usage

```python
from src.planning.a_star import a_star_search
import numpy as np

# Create a grid (0 for free space, 1 for obstacle)
grid = np.zeros((10, 10), dtype=np.int8)
grid[2:8, 5] = 1  # Add a vertical wall

# Define start and goal positions
start = (1, 1)
goal = (8, 8)

# Find the path
path, metrics = a_star_search(grid, start, goal)

# Check if a path was found
if path:
    print(f"Path found with length {len(path)}")
    print(f"Path: {path}")
    print(f"Metrics: {metrics}")
else:
    print("No path found")
```

### Advanced Options

The A* implementation includes advanced options for customizing the search:

```python
path, metrics = a_star_search(
    grid=grid,
    start=start,
    goal=goal,
    heuristic="euclidean",  # Can be "manhattan", "euclidean", "chebyshev", or a custom function
    diagonal=True,          # Allow diagonal movements
    timeout=1.0,            # Maximum time in seconds (None for no limit)
    weight=1.2              # Weight for the heuristic (> 1.0 trades optimality for speed)
)
```

### Bidirectional A* Search

For large grids, the bidirectional A* search can be more efficient:

```python
from src.planning.a_star import bidirectional_a_star

path, metrics = bidirectional_a_star(
    grid=grid,
    start=start,
    goal=goal,
    heuristic="manhattan",
    diagonal=False,
    timeout=None
)
```

### Metrics

The A* search returns detailed metrics about the search process:

- `nodes_explored`: Number of nodes expanded during the search.
- `path_length`: Length of the found path (number of positions).
- `time_taken`: Time taken to find the path in seconds.
- `path_cost`: Total cost of the path.
- `timeout`: Whether the search was terminated due to timeout (only present if timeout occurred).

## Integration with Navigation Environment

The planning module is designed to work seamlessly with the navigation environment:

```python
from src.environments.navigation_env import NavigationEnv
from src.planning.a_star import a_star_search
from src.visualization.env_visualizer import visualize_path

# Create a navigation environment
env = NavigationEnv(grid=grid, start_pos=start, goal_pos=goal)

# Find the optimal path
path, metrics = a_star_search(env.grid, env.start_pos, env.goal_pos)

# Visualize the path
visualize_path(env, path, title=f"A* Path (Length: {len(path)})")
```

## Performance Considerations

- The A* search can be memory-intensive for large grids.
- The choice of heuristic significantly affects performance.
- For real-time applications, consider using weighted A* or setting a timeout.
- The bidirectional search is generally faster for large grids but uses more memory.

## References

- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.
- Pohl, I. (1969). Bi-directional and Heuristic Search in Path Problems. Stanford University, Department of Computer Science. 