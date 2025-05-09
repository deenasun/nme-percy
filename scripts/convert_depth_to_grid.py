#!/usr/bin/env python
"""Script to convert depth maps to navigable grid environments.

This script takes a depth map (generated from the depth estimation module)
and converts it to a grid environment suitable for navigation.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.grid_conversion.converter import depth_to_grid, add_start_goal_points, save_grid, export_grid
from src.visualization.grid_visualizer import visualize_grid, visualize_depth_to_grid_comparison
from src.depth_estimation.midas import estimate_depth
from src.utils import find_start_goal_positions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Convert depth maps to navigable grid environments.")
    
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True, 
        help="Path to the input depth map (.npy file) or RGB image (will generate depth map first)."
    )
    
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=32, 
        help="Size of the output grid (grid_size x grid_size). Defaults to 32."
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.6, 
        help="Threshold for obstacle detection. Values above this threshold are considered obstacles. Range [0, 1]. Defaults to 0.6."
    )
    
    parser.add_argument(
        "--smoothing", 
        action="store_true", 
        help="Apply Gaussian smoothing to the depth map before conversion (only used for cost grids, not binary grids)."
    )
    
    parser.add_argument(
        "--kernel-size", 
        type=int, 
        default=3, 
        help="Size of the smoothing kernel if smoothing is enabled. Defaults to 3."
    )
    
    parser.add_argument(
        "--invert", 
        action="store_true", 
        help="Invert the obstacle detection (far objects become obstacles)."
    )
    
    parser.add_argument(
        "--start-pos", 
        type=str, 
        default="0,0", 
        help="Start position (row,col) for the grid environment. Defaults to '0,0'."
    )
    
    parser.add_argument(
        "--goal-pos", 
        type=str, 
        help="Goal position (row,col) for the grid environment. If not provided, defaults to the bottom-right corner."
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        default="results/grid_maps", 
        help="Directory to save the output files. Defaults to 'results/grid_maps'."
    )
    
    parser.add_argument(
        "--save-visualization", 
        action="store_true", 
        help="Save a visualization of the grid environment."
    )
    
    parser.add_argument(
        "--save-comparison", 
        action="store_true", 
        help="Save a comparison visualization of the RGB image, depth map, and grid environment."
    )
    
    parser.add_argument(
        "--no-display", 
        action="store_true", 
        help="Don't display the visualization, just save it."
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        help="Random seed for start and goal position selection."
    )
    
    parser.add_argument(
        "--results-dir", 
        type=str, 
        help="Directory to save results."
    )
    
    return parser.parse_args()


def parse_position(pos_str: str) -> tuple:
    """Parse a position string 'row,col' into a tuple of integers.
    
    Args:
        pos_str: Position string in the format 'row,col'.
        
    Returns:
        Tuple of integers (row, col).
        
    Raises:
        ValueError: If the position string is not in the correct format.
    """
    try:
        row, col = map(int, pos_str.split(','))
        return (row, col)
    except ValueError:
        raise ValueError(f"Invalid position format: {pos_str}. Expected 'row,col' with integers.")


def main() -> None:
    """Run the depth map to grid conversion."""
    args = parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load input
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Process input based on file extension
    if input_path.suffix.lower() == '.npy':
        # Load depth map directly from .npy file
        print(f"Loading depth map from {input_path}")
        try:
            depth_map = np.load(input_path)
            rgb_image = None  # No RGB image available
        except Exception as e:
            print(f"Error loading depth map: {e}")
            sys.exit(1)
    else:
        # Assume it's an RGB image and generate depth map
        print(f"Generating depth map from RGB image: {input_path}")
        try:
            rgb_image, depth_map = estimate_depth(input_path, visualize=not args.no_display)
            
            # Save the depth map for future use
            depth_path = output_dir / f"{input_path.stem}_depth.npy"
            np.save(depth_path, depth_map)
            print(f"Depth map saved to {depth_path}")
        except Exception as e:
            print(f"Error generating depth map: {e}")
            sys.exit(1)
    
    # Convert to grid
    print(f"Converting depth map to grid (size: {args.grid_size}x{args.grid_size}, threshold: {args.threshold})")
    try:
        grid = depth_to_grid(
            depth_map=depth_map,
            grid_size=args.grid_size,
            threshold=args.threshold,
            invert=args.invert,
            kernel_size=args.kernel_size
        )
        
        # Find optimal start and goal positions
        try:
            start_pos, goal_pos = find_start_goal_positions(grid)
            print(f"Automatically determined start position: {start_pos}")
            print(f"Automatically determined goal position: {goal_pos}")
        except ValueError as e:
            print(f"Error finding start/goal positions: {e}")
            print("Falling back to default positions...")
            start_pos = (0, 0)
            goal_pos = (grid.shape[0]-1, grid.shape[1]-1)
            # Ensure these positions are clear
            grid[start_pos] = 0
            grid[goal_pos] = 0
        
        # Output files
        grid_path = output_dir / f"{input_path.stem}_grid.npy"
        grid_img_path = output_dir / f"{input_path.stem}_grid.png"
        visualization_path = output_dir / f"{input_path.stem}_grid_vis.png"
        comparison_path = output_dir / f"{input_path.stem}_comparison.png"
        
        # Export grid
        export_grid(grid, grid_path)
        print(f"Grid exported to {grid_path}")
        
        # Save grid as image
        save_grid(grid, grid_img_path)
        print(f"Grid image saved to {grid_img_path}")
        
        # Visualize grid
        if args.save_visualization:
            visualize_grid(
                grid=grid,
                title=f"Grid Environment (size: {args.grid_size}x{args.grid_size})",
                save_path=visualization_path,
                show=not args.no_display
            )
            print(f"Grid visualization saved to {visualization_path}")
        
        # Visualize comparison
        if args.save_comparison and rgb_image is not None:
            visualize_depth_to_grid_comparison(
                rgb_image=rgb_image,
                depth_map=depth_map,
                grid=grid,
                save_path=comparison_path,
                show=not args.no_display
            )
            print(f"Comparison visualization saved to {comparison_path}")
        
        print("Conversion complete!")
        
    except Exception as e:
        print(f"Error during grid conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 