"""Depth map to grid environment conversion module.

This module provides functionality to convert continuous depth maps 
into discrete grid representations suitable for navigation.
"""

from typing import Tuple, Optional, Union, Literal
import os
import logging
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage import median_filter

from src.depth_estimation.utils import normalize_depth_map

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Grid cell types
FREE_SPACE = 0
OBSTACLE = 1
START_POINT = 2
GOAL_POINT = 3


def depth_to_grid(
    depth_map: np.ndarray,
    grid_size: int = 20,
    threshold: float = 0.5,
    invert: bool = False,
    kernel_size: int = 3
) -> np.ndarray:
    """Convert a depth map to a navigable grid environment.
    
    Args:
        depth_map: The input depth map (2D numpy array).
        grid_size: Size of the output grid (grid_size x grid_size).
        threshold: Threshold for determining obstacles (0.0 to 1.0).
        invert: Whether to invert the depth map (darker is farther).
        kernel_size: Size of the median filter kernel for noise reduction.
        
    Returns:
        A 2D grid environment as a numpy array where:
            0 = free space
            1 = obstacle
    
    Raises:
        ValueError: If grid_size is not positive or threshold is not in range [0, 1].
    """
    if grid_size <= 0:
        raise ValueError(f"Grid size must be positive, got {grid_size}")
    
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be in range [0, 1], got {threshold}")
    
    # Normalize depth map to [0, 1]
    logger.info("Normalizing depth map")
    depth_normalized = depth_map.copy().astype(np.float32)
    if np.max(depth_normalized) > 0:  # Avoid division by zero
        depth_normalized = (depth_normalized - np.min(depth_normalized)) / (np.max(depth_normalized) - np.min(depth_normalized))
    
    # Invert if needed (1 is close, 0 is far)
    if invert:
        depth_normalized = 1.0 - depth_normalized
    
    # Resize to grid size
    logger.info(f"Resizing depth map to grid size {grid_size}x{grid_size}")
    depth_resized = resize(depth_normalized, (grid_size, grid_size), anti_aliasing=True)
    
    # Apply median filter to reduce noise
    if kernel_size > 1:
        logger.info(f"Applying median filter with kernel size {kernel_size}")
        depth_filtered = median_filter(depth_resized, size=kernel_size)
    else:
        depth_filtered = depth_resized
    
    # Threshold to create binary grid
    grid = (depth_filtered > threshold).astype(np.int8)
    
    logger.info(f"Grid conversion complete. Obstacle ratio: {np.mean(grid):.2f}")
    return grid


def depth_to_cost_grid(
    depth_map: np.ndarray,
    grid_size: int = 32,
    min_cost: int = 1,
    max_cost: int = 100,
    invert: bool = False,
    smoothing: bool = True,
    kernel_size: int = 3
) -> np.ndarray:
    """Convert a depth map to a cost grid where each cell has a traversal cost.
    
    This creates a more nuanced representation than the binary grid, where 
    the cost of traversing a cell is proportional to its depth value.
    
    Args:
        depth_map: The input depth map (2D numpy array).
        grid_size: Size of the output grid (grid_size x grid_size).
        min_cost: Minimum cost value in the output grid.
        max_cost: Maximum cost value in the output grid.
        invert: If True, inverts the cost mapping (far objects have higher cost).
        smoothing: Whether to apply Gaussian smoothing to the depth map before conversion.
        kernel_size: Size of the smoothing kernel if smoothing is enabled.
        
    Returns:
        A 2D cost grid as a numpy array where each cell contains a cost value
        between min_cost and max_cost.
    
    Raises:
        ValueError: If grid_size is not positive or min_cost > max_cost.
    """
    if grid_size <= 0:
        raise ValueError(f"Grid size must be positive, got {grid_size}")
        
    if min_cost > max_cost:
        raise ValueError(f"min_cost ({min_cost}) must be less than or equal to max_cost ({max_cost})")
    
    # Normalize depth map to [0, 1]
    logger.info("Normalizing depth map")
    depth_norm = normalize_depth_map(depth_map)
    
    # Apply Gaussian smoothing if enabled
    if smoothing:
        logger.info(f"Applying Gaussian smoothing with kernel size {kernel_size}")
        depth_norm = cv2.GaussianBlur(depth_norm, (kernel_size, kernel_size), 0)
    
    # Resize to desired grid size
    logger.info(f"Resizing depth map to grid size {grid_size}x{grid_size}")
    depth_resized = cv2.resize(depth_norm, (grid_size, grid_size))
    
    # Map depth values to cost range
    if invert:
        # Far objects have higher cost
        cost_grid = (1 - depth_resized) * (max_cost - min_cost) + min_cost
    else:
        # Close objects have higher cost
        cost_grid = depth_resized * (max_cost - min_cost) + min_cost
    
    # Convert to integer
    cost_grid = cost_grid.astype(np.int32)
    
    logger.info(f"Cost grid conversion complete. Average cost: {np.mean(cost_grid):.2f}")
    return cost_grid


def add_start_goal_points(
    grid: np.ndarray,
    start_pos: Tuple[int, int] = (0, 0),
    goal_pos: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Add start and goal points to the grid.
    
    Args:
        grid: The input grid (2D numpy array).
        start_pos: Tuple (row, col) for the start position.
        goal_pos: Tuple (row, col) for the goal position. If None, 
            defaults to the bottom-right corner.
            
    Returns:
        Grid with start and goal points marked as 2 and 3 respectively.
        
    Raises:
        ValueError: If start_pos or goal_pos are outside the grid.
    """
    # Create a copy of the grid to avoid modifying the original
    grid_with_points = grid.copy()
    
    # Set default goal position to bottom-right if not specified
    if goal_pos is None:
        goal_pos = (grid.shape[0] - 1, grid.shape[1] - 1)
    
    # Validate positions
    grid_shape = grid.shape
    if not (0 <= start_pos[0] < grid_shape[0] and 0 <= start_pos[1] < grid_shape[1]):
        raise ValueError(f"Start position {start_pos} is outside grid of shape {grid_shape}")
    
    if not (0 <= goal_pos[0] < grid_shape[0] and 0 <= goal_pos[1] < grid_shape[1]):
        raise ValueError(f"Goal position {goal_pos} is outside grid of shape {grid_shape}")
    
    # Mark start and goal positions
    grid_with_points[start_pos] = START_POINT
    grid_with_points[goal_pos] = GOAL_POINT
    
    return grid_with_points


def save_grid(
    grid: np.ndarray,
    output_path: Union[str, Path],
    colormap: bool = True
) -> None:
    """Save grid to an image file.
    
    Args:
        grid: The grid to save (2D numpy array).
        output_path: Path where the grid will be saved.
        colormap: Whether to apply a colormap for visualization.
            If True, uses a distinct color for each cell type.
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if colormap:
        # Create a colored visualization
        # Define colors for each cell type (in BGR for OpenCV)
        colors = {
            FREE_SPACE: (255, 255, 255),  # White for free space
            OBSTACLE: (0, 0, 0),          # Black for obstacles
            START_POINT: (0, 255, 0),     # Green for start
            GOAL_POINT: (0, 0, 255)       # Red for goal
        }
        
        # Create RGB image
        grid_vis = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        
        # Set colors for each cell type
        for cell_type, color in colors.items():
            grid_vis[grid == cell_type] = color
            
        # Save the visualization
        cv2.imwrite(str(output_path), grid_vis)
    else:
        # Save as grayscale
        cv2.imwrite(str(output_path), grid.astype(np.uint8) * 85)  # Scale to be visible
    
    logger.info(f"Grid saved to {output_path}")


def export_grid(
    grid: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """Export grid as a numpy array.
    
    Args:
        grid: The grid to export (2D numpy array).
        output_path: Path where the grid will be saved.
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as numpy array
    np.save(output_path, grid)
    logger.info(f"Grid exported to {output_path}")


def visualize_grid(
    grid: np.ndarray,
    start_pos: Optional[Tuple[int, int]] = None,
    goal_pos: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Visualize a grid environment.
    
    Args:
        grid: The grid to visualize (2D numpy array).
        start_pos: Tuple (row, col) for the start position.
        goal_pos: Tuple (row, col) for the goal position.
        save_path: Path to save the visualization.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary', interpolation='nearest')
    
    if start_pos:
        plt.plot(start_pos[1], start_pos[0], 'go', markersize=10, label='Start')
    
    if goal_pos:
        plt.plot(goal_pos[1], goal_pos[0], 'ro', markersize=10, label='Goal')
    
    plt.title('Grid Environment')
    plt.legend()
    plt.grid(False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close() 