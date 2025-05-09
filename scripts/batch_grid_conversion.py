#!/usr/bin/env python
"""Batch script for converting multiple depth maps to grid environments.

This script processes multiple depth maps (or RGB images) and converts them
to grid environments suitable for navigation.
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from typing import List, Tuple

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.grid_conversion.converter import depth_to_grid, add_start_goal_points, save_grid, export_grid
from src.visualization.grid_visualizer import visualize_grid, visualize_depth_to_grid_comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Batch conversion of depth maps to grid environments.")
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory containing input depth maps (.npy files) or RGB images."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/grid_maps",
        help="Directory to save the output files. Defaults to 'results/grid_maps'."
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern to match specific files in the input directory. Default is '*'."
    )
    
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["depth", "rgb", "both"],
        default="both",
        help="Type of input files to process: 'depth' for .npy depth maps only, 'rgb' for RGB images only, or 'both' for all files. Default is 'both'."
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
        help="Threshold factor for obstacle detection. Values below this threshold are considered obstacles. Range [0, 1]. Defaults to 0.6."
    )
    
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="Apply Gaussian smoothing to the depth map before conversion."
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
        "--save-visualization",
        action="store_true",
        help="Save visualizations of the grid environments."
    )
    
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save comparison visualizations of RGB images, depth maps, and grid environments (only for RGB inputs)."
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display any visualizations, just save them."
    )
    
    return parser.parse_args()


def parse_position(pos_str: str) -> Tuple[int, int]:
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


def find_input_files(input_dir: str, pattern: str, input_type: str) -> Tuple[List[str], List[str]]:
    """Find depth map files and RGB image files in the input directory.
    
    Args:
        input_dir: Directory to search for input files.
        pattern: Glob pattern to match specific files.
        input_type: Type of input files to find: 'depth', 'rgb', or 'both'.
        
    Returns:
        A tuple containing:
            - List of paths to depth map files (.npy)
            - List of paths to RGB image files
    """
    depth_files = []
    rgb_files = []
    
    # Combine directory path with pattern
    search_pattern = os.path.join(input_dir, pattern)
    
    # Find all matching files
    all_files = glob.glob(search_pattern)
    
    # Separate depth maps and RGB images
    for file_path in all_files:
        if file_path.lower().endswith('.npy') and (input_type == 'depth' or input_type == 'both'):
            depth_files.append(file_path)
        elif any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']) and (input_type == 'rgb' or input_type == 'both'):
            rgb_files.append(file_path)
    
    return depth_files, rgb_files


def process_depth_map(
    depth_path: str,
    output_dir: Path,
    grid_size: int,
    threshold: float,
    smoothing: bool,
    kernel_size: int,
    invert: bool,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int] = None,
    save_visualization: bool = False,
    no_display: bool = False
) -> None:
    """Process a single depth map file.
    
    Args:
        depth_path: Path to the depth map file.
        output_dir: Directory to save output files.
        grid_size: Size of the output grid.
        threshold: Threshold factor for obstacle detection.
        smoothing: Whether to apply smoothing.
        kernel_size: Size of the smoothing kernel.
        invert: Whether to invert obstacle detection.
        start_pos: Start position (row, col).
        goal_pos: Goal position (row, col).
        save_visualization: Whether to save visualizations.
        no_display: Whether to suppress display of visualizations.
    """
    try:
        # Load depth map
        print(f"Processing depth map: {depth_path}")
        depth_map = np.load(depth_path)
        
        # Get base filename
        file_stem = Path(depth_path).stem
        
        # Convert to grid
        grid = depth_to_grid(
            depth_map=depth_map,
            grid_size=grid_size,
            threshold_factor=threshold,
            smoothing=smoothing,
            kernel_size=kernel_size,
            invert=invert
        )
        
        # Add start and goal points
        grid_with_points = add_start_goal_points(grid, start_pos, goal_pos)
        
        # Output paths
        grid_path = output_dir / f"{file_stem}_grid.npy"
        grid_img_path = output_dir / f"{file_stem}_grid.png"
        vis_path = output_dir / f"{file_stem}_grid_vis.png" if save_visualization else None
        
        # Export grid
        export_grid(grid, grid_path)
        
        # Save grid image
        save_grid(grid_with_points, grid_img_path)
        
        # Save visualization if requested
        if save_visualization:
            visualize_grid(
                grid=grid_with_points,
                title=f"Grid Environment from {file_stem}",
                save_path=vis_path,
                show=not no_display
            )
        
        print(f"  - Processed successfully")
        
    except Exception as e:
        print(f"  - Error processing {depth_path}: {e}")


def process_rgb_image(
    image_path: str,
    output_dir: Path,
    grid_size: int,
    threshold: float,
    smoothing: bool,
    kernel_size: int,
    invert: bool,
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int] = None,
    save_visualization: bool = False,
    save_comparison: bool = False,
    no_display: bool = False
) -> None:
    """Process a single RGB image file.
    
    Args:
        image_path: Path to the RGB image file.
        output_dir: Directory to save output files.
        grid_size: Size of the output grid.
        threshold: Threshold factor for obstacle detection.
        smoothing: Whether to apply smoothing.
        kernel_size: Size of the smoothing kernel.
        invert: Whether to invert obstacle detection.
        start_pos: Start position (row, col).
        goal_pos: Goal position (row, col).
        save_visualization: Whether to save visualizations.
        save_comparison: Whether to save comparison visualizations.
        no_display: Whether to suppress display of visualizations.
    """
    try:
        # Get base filename
        file_stem = Path(image_path).stem
        print(f"Processing RGB image: {image_path}")
        
        # Generate depth map
        from src.depth_estimation.midas import estimate_depth
        rgb_image, depth_map = estimate_depth(image_path, visualize=False)
        
        # Save depth map
        depth_path = output_dir / f"{file_stem}_depth.npy"
        np.save(depth_path, depth_map)
        
        # Convert to grid
        grid = depth_to_grid(
            depth_map=depth_map,
            grid_size=grid_size,
            threshold_factor=threshold,
            smoothing=smoothing,
            kernel_size=kernel_size,
            invert=invert
        )
        
        # Add start and goal points
        grid_with_points = add_start_goal_points(grid, start_pos, goal_pos)
        
        # Output paths
        grid_path = output_dir / f"{file_stem}_grid.npy"
        grid_img_path = output_dir / f"{file_stem}_grid.png"
        vis_path = output_dir / f"{file_stem}_grid_vis.png" if save_visualization else None
        comparison_path = output_dir / f"{file_stem}_comparison.png" if save_comparison else None
        
        # Export grid
        export_grid(grid, grid_path)
        
        # Save grid image
        save_grid(grid_with_points, grid_img_path)
        
        # Save visualization if requested
        if save_visualization:
            visualize_grid(
                grid=grid_with_points,
                title=f"Grid Environment from {file_stem}",
                save_path=vis_path,
                show=not no_display
            )
        
        # Save comparison if requested
        if save_comparison:
            visualize_depth_to_grid_comparison(
                rgb_image=rgb_image,
                depth_map=depth_map,
                grid=grid_with_points,
                title=f"From RGB to Depth to Grid: {file_stem}",
                save_path=comparison_path,
                show=not no_display
            )
        
        print(f"  - Processed successfully")
        
    except Exception as e:
        print(f"  - Error processing {image_path}: {e}")


def main() -> None:
    """Run batch conversion of depth maps to grid environments."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse start and goal positions
    start_pos = parse_position(args.start_pos)
    goal_pos = parse_position(args.goal_pos) if args.goal_pos else None
    
    # Find input files
    depth_files, rgb_files = find_input_files(args.input_dir, args.pattern, args.input_type)
    
    # Report findings
    print(f"Found {len(depth_files)} depth map files and {len(rgb_files)} RGB image files to process")
    
    # Process depth maps
    for depth_path in depth_files:
        process_depth_map(
            depth_path=depth_path,
            output_dir=output_dir,
            grid_size=args.grid_size,
            threshold=args.threshold,
            smoothing=args.smoothing,
            kernel_size=args.kernel_size,
            invert=args.invert,
            start_pos=start_pos,
            goal_pos=goal_pos,
            save_visualization=args.save_visualization,
            no_display=args.no_display
        )
    
    # Process RGB images
    for image_path in rgb_files:
        process_rgb_image(
            image_path=image_path,
            output_dir=output_dir,
            grid_size=args.grid_size,
            threshold=args.threshold,
            smoothing=args.smoothing,
            kernel_size=args.kernel_size,
            invert=args.invert,
            start_pos=start_pos,
            goal_pos=goal_pos,
            save_visualization=args.save_visualization,
            save_comparison=args.save_comparison,
            no_display=args.no_display
        )
    
    print(f"Batch processing complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main() 