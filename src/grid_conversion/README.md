# Grid Conversion Module

This module provides functionality to convert depth maps into navigable grid environments suitable for reinforcement learning and path planning.

## Overview

The grid conversion module consists of the following components:

- `converter.py`: Core functionality for converting depth maps to grid environments
- Grid cell type constants (`FREE_SPACE`, `OBSTACLE`, `START_POINT`, `GOAL_POINT`)

## Functions

### Basic Conversion

```python
from src.grid_conversion.converter import depth_to_grid

# Convert depth map to grid environment
grid = depth_to_grid(
    depth_map=depth_map,
    grid_size=32,
    threshold_factor=0.6,
    smoothing=True,
    kernel_size=3,
    invert=False
)
```

### Cost Grid

For more nuanced navigation, you can create a cost grid where each cell has a traversal cost:

```python
from src.grid_conversion.converter import depth_to_cost_grid

# Convert depth map to cost grid
cost_grid = depth_to_cost_grid(
    depth_map=depth_map,
    grid_size=32,
    min_cost=1,
    max_cost=100,
    invert=False
)
```

### Adding Start and Goal Points

```python
from src.grid_conversion.converter import add_start_goal_points

# Add start and goal points to grid
grid_with_points = add_start_goal_points(
    grid=grid,
    start_pos=(0, 0),
    goal_pos=(31, 31)
)
```

### Saving and Exporting

```python
from src.grid_conversion.converter import save_grid, export_grid

# Save grid as image
save_grid(grid_with_points, "grid.png", colormap=True)

# Export grid as numpy array
export_grid(grid, "grid.npy")
```

## Cell Types

The module defines the following cell types:

- `FREE_SPACE` (0): Navigable space
- `OBSTACLE` (1): Non-navigable obstacle
- `START_POINT` (2): Starting position
- `GOAL_POINT` (3): Goal position

## Parameters

### depth_to_grid

- `depth_map`: The input depth map (2D numpy array)
- `grid_size`: Size of the output grid (grid_size x grid_size)
- `threshold_factor`: Threshold for obstacle detection (0-1)
- `smoothing`: Whether to apply Gaussian smoothing
- `kernel_size`: Size of the smoothing kernel
- `invert`: Whether to invert obstacle detection

### depth_to_cost_grid

- `depth_map`: The input depth map (2D numpy array)
- `grid_size`: Size of the output grid (grid_size x grid_size)
- `min_cost`: Minimum cost value in the output grid
- `max_cost`: Maximum cost value in the output grid
- `invert`: Whether to invert the cost mapping
- `smoothing`: Whether to apply Gaussian smoothing
- `kernel_size`: Size of the smoothing kernel

## Visualization

For visualization of grid environments, see the `src/visualization/grid_visualizer.py` module. 