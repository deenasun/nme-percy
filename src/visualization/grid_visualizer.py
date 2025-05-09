"""Grid visualization utilities.

This module provides functions for visualizing grid environments.
"""

from typing import Tuple, Optional, Union, List, Dict
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import cv2

from src.grid_conversion.converter import FREE_SPACE, OBSTACLE, START_POINT, GOAL_POINT


def visualize_grid(
    grid: np.ndarray,
    title: str = "Grid Environment",
    cmap: Optional[str] = None,
    show_grid_lines: bool = True,
    fig_size: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Visualize a grid environment.
    
    Args:
        grid: The grid to visualize (2D numpy array).
        title: Title for the plot.
        cmap: Colormap to use. If None, uses a custom colormap for grid values.
        show_grid_lines: Whether to show grid lines.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the visualization. If None, won't save.
        show: Whether to display the plot.
        dpi: DPI for the saved figure.
    """
    # Create a custom colormap if none provided
    if cmap is None:
        # Define colors for each cell type
        colors = ["white", "black", "green", "red"]
        cmap = mcolors.ListedColormap(colors)
        
        # Set color limits
        vmin, vmax = 0, 3
    else:
        # Use provided colormap with data limits
        vmin, vmax = None, None
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Plot grid
    plt.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add title
    plt.title(title)
    
    # Show grid lines if requested
    if show_grid_lines and grid.shape[0] <= 50:  # Only show for smaller grids
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    
    # Add a colorbar for non-custom colormaps
    if cmap is not None and cmap != "custom":
        plt.colorbar()
    
    # Save visualization if requested
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()


def visualize_grid_with_path(
    grid: np.ndarray,
    path: List[Tuple[int, int]],
    title: str = "Path in Grid Environment",
    path_color: str = "blue",
    path_width: float = 2.0,
    show_grid_lines: bool = True,
    fig_size: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Visualize a grid environment with a path.
    
    Args:
        grid: The grid to visualize (2D numpy array).
        path: List of (row, col) coordinates defining the path.
        title: Title for the plot.
        path_color: Color of the path.
        path_width: Width of the path line.
        show_grid_lines: Whether to show grid lines.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the visualization. If None, won't save.
        show: Whether to display the plot.
        dpi: DPI for the saved figure.
    """
    # Create a custom colormap
    colors = ["white", "black", "green", "red"]
    cmap = mcolors.ListedColormap(colors)
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Plot grid
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    
    # Add title
    plt.title(title)
    
    # Show grid lines if requested
    if show_grid_lines and grid.shape[0] <= 50:  # Only show for smaller grids
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    
    # Plot path
    if path:
        # Extract row and column indices
        rows, cols = zip(*path)
        
        # Plot path
        plt.plot(cols, rows, color=path_color, linewidth=path_width, marker='o', markersize=4)
    
    # Save visualization if requested
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()


def visualize_depth_to_grid_comparison(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    grid: np.ndarray,
    title: str = "From RGB to Depth to Grid",
    fig_size: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 150
) -> None:
    """Visualize the transformation from RGB to depth map to grid.
    
    Args:
        rgb_image: The original RGB image.
        depth_map: The depth map generated from the RGB image.
        grid: The grid environment generated from the depth map.
        title: Title for the plot.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the visualization. If None, won't save.
        show: Whether to display the plot.
        dpi: DPI for the saved figure.
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size)
    
    # Plot RGB image
    ax1.imshow(rgb_image)
    ax1.set_title("RGB Image")
    ax1.axis("off")
    
    # Plot depth map
    ax2.imshow(depth_map, cmap="inferno")
    ax2.set_title("Depth Map")
    ax2.axis("off")
    
    # Plot grid
    # Create a custom colormap
    colors = ["white", "black", "green", "red"]
    cmap = mcolors.ListedColormap(colors)
    
    ax3.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    ax3.set_title("Grid Environment")
    
    # Show grid lines for smaller grids
    if grid.shape[0] <= 50:
        ax3.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax3.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax3.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax3.set_xticks([])
        ax3.set_yticks([])
    else:
        ax3.axis("off")
    
    # Add title to the figure
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save visualization if requested
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close() 