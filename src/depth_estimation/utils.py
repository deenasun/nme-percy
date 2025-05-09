"""Utility functions for depth map processing and export."""

from typing import Optional, Union, Tuple
import os
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize_depth_map(
    depth_map: np.ndarray, 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None
) -> np.ndarray:
    """Normalize depth map to range [0, 1].
    
    Args:
        depth_map: The depth map to normalize.
        min_val: Optional minimum value for normalization. If None, uses depth_map.min().
        max_val: Optional maximum value for normalization. If None, uses depth_map.max().
        
    Returns:
        Normalized depth map with values in range [0, 1].
    """
    min_val = min_val if min_val is not None else depth_map.min()
    max_val = max_val if max_val is not None else depth_map.max()
    
    # Prevent division by zero
    if max_val == min_val:
        return np.zeros_like(depth_map)
    
    # Normalize to [0, 1]
    normalized = (depth_map - min_val) / (max_val - min_val)
    return normalized

def convert_to_uint8(
    depth_map: np.ndarray, 
    normalize: bool = True
) -> np.ndarray:
    """Convert depth map to uint8 format (0-255).
    
    Args:
        depth_map: The depth map to convert.
        normalize: Whether to normalize the depth map before conversion.
        
    Returns:
        Depth map as uint8 array with values from 0-255.
    """
    if normalize:
        depth_map = normalize_depth_map(depth_map)
    
    # Scale to 0-255 and convert to uint8
    uint8_depth = (depth_map * 255).astype(np.uint8)
    return uint8_depth

def save_depth_map(
    depth_map: np.ndarray, 
    output_path: Union[str, Path],
    normalize: bool = True,
    colormap: Optional[int] = cv2.COLORMAP_INFERNO
) -> None:
    """Save depth map to file.
    
    Args:
        depth_map: The depth map to save.
        output_path: Path where the depth map will be saved.
        normalize: Whether to normalize the depth map before saving.
        colormap: OpenCV colormap to apply. If None, saves as grayscale.
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to uint8
    uint8_depth = convert_to_uint8(depth_map, normalize)
    
    # Apply colormap if specified
    if colormap is not None:
        uint8_depth = cv2.applyColorMap(uint8_depth, colormap)
    
    # Save image
    cv2.imwrite(str(output_path), uint8_depth)

def export_depth_data(
    depth_map: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """Export raw depth data as numpy array.
    
    Args:
        depth_map: The depth map to export.
        output_path: Path where the depth data will be saved.
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save numpy array
    np.save(output_path, depth_map)

def save_depth_visualization(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    output_path: Union[str, Path],
    colormap: str = "inferno",
    dpi: int = 150
) -> None:
    """Create and save a visualization of RGB image and depth map side by side.
    
    Args:
        rgb_image: Original RGB image.
        depth_map: Estimated depth map.
        output_path: Path where the visualization will be saved.
        colormap: Matplotlib colormap name.
        dpi: DPI for the output figure.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original RGB image
    ax1.imshow(rgb_image)
    ax1.set_title("RGB Image")
    ax1.axis("off")
    
    # Depth map
    ax2.imshow(depth_map, cmap=colormap)
    ax2.set_title("Depth Map")
    ax2.axis("off")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close() 