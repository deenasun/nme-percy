#!/usr/bin/env python
"""Script to test the A* path planning algorithm.

This script demonstrates how to use the A* search algorithm to find optimal paths
through grid environments and visualize the results.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import zoom

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.planning.a_star import a_star_search, bidirectional_a_star
from src.environments.navigation_env import NavigationEnv
from src.visualization.env_visualizer import visualize_path, visualize_trajectory
from src.grid_conversion.converter import depth_to_grid
from src.depth_estimation.midas import estimate_depth
from src.visualization.grid_visualizer import visualize_grid_with_path
from src.utils import find_start_goal_positions


def resize_grid(grid, max_size):
    """
    Resize grid to max_size x max_size if it's larger.
    Preserves the general structure while making it suitable for pathfinding visualization.
    
    Args:
        grid: Original grid as numpy array
        max_size: Maximum size for each dimension
        
    Returns:
        Resized grid
    """
    # If grid is already small enough, return as is
    if grid.shape[0] <= max_size and grid.shape[1] <= max_size:
        return grid
    
    # Calculate zoom factors to resize to max_size
    zoom_factors = [max_size / grid.shape[0], max_size / grid.shape[1]]
    
    # Use nearest-neighbor interpolation to preserve 0s and 1s
    resized_grid = zoom(grid, zoom_factors, order=0)
    
    # Ensure the output has proper binary values (0s and 1s)
    return resized_grid.astype(np.int8)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Test A* path planning algorithm.")
    
    parser.add_argument(
        "--grid", "-g",
        type=str,
        help="Path to a grid file (.npy) to use as the environment."
    )
    
    parser.add_argument(
        "--depth", "-d",
        type=str,
        help="Path to a depth map file (.npy) to convert to a grid environment."
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to an image to estimate depth and convert to a grid environment."
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=32,
        help="Size of the grid if generating. Defaults to 32."
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Threshold for obstacle detection if generating from depth map. Defaults to 0.6."
    )
    
    parser.add_argument(
        "--invert-grid",
        action="store_true",
        help="Invert the grid values (0→1, 1→0) to make dark areas navigable and light areas obstacles."
    )
    
    parser.add_argument(
        "--max-grid-size",
        type=int,
        default=256,
        help="Maximum grid size for path planning visualization (larger grids will be resized)."
    )
    
    parser.add_argument(
        "--start-pos",
        type=str,
        default="0,0",
        help="Start position as 'row,col'. Defaults to '0,0'."
    )
    
    parser.add_argument(
        "--goal-pos",
        type=str,
        help="Goal position as 'row,col'. If not provided, uses the bottom-right corner."
    )
    
    parser.add_argument(
        "--heuristic",
        type=str,
        choices=["manhattan", "euclidean", "chebyshev"],
        default="manhattan",
        help="Heuristic function to use for A*. Defaults to 'manhattan'."
    )
    
    parser.add_argument(
        "--diagonal",
        action="store_true",
        help="Allow diagonal movements in path planning."
    )
    
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional A* search instead of regular A*."
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        help="Maximum time in seconds to run the algorithm. If not provided, no timeout."
    )
    
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Weight for the heuristic (> 1.0 trades optimality for speed). Defaults to 1.0."
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different heuristics and algorithms."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/planning",
        help="Directory to save output files. Defaults to 'results/planning'."
    )
    
    parser.add_argument(
        "--show-explored",
        action="store_true",
        help="Show explored nodes in the grid visualization."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file name for the visualization."
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display the visualization."
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


def create_test_grid(grid_size: int = 32, pattern: str = "maze") -> np.ndarray:
    """Create a test grid with a specific pattern.
    
    Args:
        grid_size: Size of the grid (height and width).
        pattern: Type of pattern to create:
            - "maze": A simple maze pattern.
            - "random": Random obstacles.
            - "empty": Empty grid with walls along the edges.
            
    Returns:
        A 2D numpy array representing the grid (0 for free space, 1 for obstacle).
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    
    if pattern == "empty":
        # Add walls along the edges
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
    
    elif pattern == "random":
        # Add walls along the edges
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        # Add random obstacles (about 30% of interior cells)
        np.random.seed(42)  # For reproducibility
        for _ in range(int(0.3 * (grid_size - 2) ** 2)):
            row = np.random.randint(1, grid_size - 1)
            col = np.random.randint(1, grid_size - 1)
            grid[row, col] = 1
    
    elif pattern == "maze":
        # Create a simple maze pattern
        # Start with walls everywhere
        grid[:] = 1
        
        # Clear the center
        center = grid_size // 2
        radius = grid_size // 4
        for r in range(center - radius, center + radius):
            for c in range(center - radius, center + radius):
                grid[r, c] = 0
        
        # Add some paths
        # Horizontal paths
        for i in range(0, grid_size, 4):
            grid[i, 1:-1] = 0
        
        # Vertical paths
        for i in range(0, grid_size, 4):
            grid[1:-1, i] = 0
        
        # Add some random connections
        np.random.seed(42)
        for _ in range(grid_size):
            r = np.random.randint(1, grid_size - 1)
            c = np.random.randint(1, grid_size - 1)
            size = np.random.randint(3, 7)
            if np.random.rand() > 0.5:
                # Horizontal
                grid[r, max(1, c - size):min(grid_size - 1, c + size)] = 0
            else:
                # Vertical
                grid[max(1, r - size):min(grid_size - 1, r + size), c] = 0
    
    # Make sure the corners are free for start and goal positions
    # Top-left corner (typical start)
    grid[1, 1] = 0
    grid[1, 2] = 0
    grid[2, 1] = 0
    
    # Bottom-right corner (typical goal)
    grid[-2, -2] = 0
    grid[-2, -3] = 0
    grid[-3, -2] = 0
    
    return grid


def load_or_generate_grid(args: argparse.Namespace) -> np.ndarray:
    """Load a grid from a file or generate one.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        A 2D numpy array representing the grid environment.
    """
    if args.grid:
        # Load grid from file
        print(f"Loading grid from {args.grid}")
        grid = np.load(args.grid)
    elif args.depth:
        # Generate grid from depth map
        print(f"Generating grid from depth map {args.depth}")
        depth_map = np.load(args.depth)
        
        grid = depth_to_grid(
            depth_map=depth_map,
            grid_size=args.grid_size,
            threshold_factor=args.threshold,
            smoothing=True,
            kernel_size=3
        )
    elif args.image:
        # Estimate depth from image and convert to grid
        print(f"Estimating depth from image {args.image}")
        _, depth_map = estimate_depth(args.image)
        grid = depth_to_grid(depth_map, grid_size=args.grid_size)
    else:
        # Generate a test grid
        print(f"Generating a test grid with size {args.grid_size}x{args.grid_size}")
        grid = create_test_grid(args.grid_size, pattern="maze")
    
    return grid


def compare_algorithms(
    grid: np.ndarray, 
    start: tuple, 
    goal: tuple, 
    output_dir: Path,
    diagonal: bool = False,
    timeout: float = None
) -> None:
    """Compare different path planning algorithms and heuristics.
    
    Args:
        grid: The grid environment.
        start: Starting position.
        goal: Goal position.
        output_dir: Directory to save comparison results.
        diagonal: Whether to allow diagonal movements.
        timeout: Maximum time in seconds for each algorithm.
    """
    # Create a figure for the comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Flattened axes for easy indexing
    axes = axes.flatten()
    
    # Create navigation environment for visualization
    env = NavigationEnv(
        grid=grid,
        start_pos=start,
        goal_pos=goal,
        render_mode=None
    )
    
    # Dictionary to store results
    results = {}
    
    # Test different algorithms and heuristics
    algorithms = [
        ("A* (Manhattan)", a_star_search, "manhattan", 1.0),
        ("A* (Euclidean)", a_star_search, "euclidean", 1.0),
        ("A* (Chebyshev)", a_star_search, "chebyshev", 1.0),
        ("Bidirectional A*", bidirectional_a_star, "manhattan", None)
    ]
    
    for i, (name, algorithm, heuristic, weight) in enumerate(algorithms):
        print(f"\nRunning {name}...")
        start_time = time.time()
        
        # Run the algorithm
        if algorithm == a_star_search:
            path, metrics = algorithm(
                grid=grid,
                start=start,
                goal=goal,
                heuristic=heuristic,
                diagonal=diagonal,
                timeout=timeout,
                weight=weight
            )
        else:
            path, metrics = algorithm(
                grid=grid,
                start=start,
                goal=goal,
                heuristic=heuristic,
                diagonal=diagonal,
                timeout=timeout
            )
        
        # Store results
        results[name] = {
            "path": path,
            "metrics": metrics
        }
        
        # Print results
        if path:
            print(f"  Path found with length {len(path)}")
            print(f"  Nodes explored: {metrics['nodes_explored']}")
            print(f"  Time taken: {metrics['time_taken']:.4f} seconds")
        else:
            print("  No path found")
        
        # Create a subplot for this algorithm
        ax = axes[i]
        
        # Visualize the grid
        vis_grid = grid.copy()
        vis_grid[start] = 2  # Start
        vis_grid[goal] = 3   # Goal
        
        # Create a colormap for visualization
        colors = ["white", "black", "green", "red"]
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        # Show the grid
        ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=3)
        
        # Add title with results
        title = f"{name}\n"
        if path:
            title += f"Path Length: {len(path)}, "
            title += f"Nodes Explored: {metrics['nodes_explored']}, "
            title += f"Time: {metrics['time_taken']:.4f}s"
        else:
            title += "No path found"
        
        ax.set_title(title)
        
        # Plot the path if found
        if path:
            # Extract row and column indices
            rows, cols = zip(*path)
            ax.plot(cols, rows, 'b-', linewidth=2)
            ax.plot(cols, rows, 'bo', markersize=4)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout and save the figure
    plt.tight_layout()
    comparison_path = output_dir / "algorithm_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    print(f"\nComparison saved to {comparison_path}")
    
    # Create a table of results
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    
    # Create table data
    table_data = [
        ["Algorithm", "Path Length", "Nodes Explored", "Time (s)", "Path Cost"]
    ]
    
    for name in results:
        metrics = results[name]["metrics"]
        path = results[name]["path"]
        
        row = [
            name,
            str(len(path)) if path else "N/A",
            str(metrics["nodes_explored"]),
            f"{metrics['time_taken']:.4f}",
            f"{metrics['path_cost']:.1f}" if "path_cost" in metrics else "N/A"
        ]
        
        table_data.append(row)
    
    # Create table
    table = plt.table(
        cellText=table_data,
        colWidths=[0.25, 0.15, 0.2, 0.15, 0.15],
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Highlight header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white')
    
    # Save the table
    table_path = output_dir / "algorithm_comparison_table.png"
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    print(f"Comparison table saved to {table_path}")


def main() -> None:
    """Main function to test the A* search algorithm."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process input and create a grid
    if args.grid:
        print(f"Loading grid from {args.grid}")
        grid = np.load(args.grid)
    elif args.depth:
        print(f"Loading depth map from {args.depth}")
        depth_map = np.load(args.depth)
        grid = depth_to_grid(depth_map, grid_size=args.grid_size)
    elif args.image:
        print(f"Estimating depth from image {args.image}")
        _, depth_map = estimate_depth(args.image)
        grid = depth_to_grid(depth_map, grid_size=args.grid_size)
    else:
        print(f"Creating random grid of size {args.grid_size}x{args.grid_size}")
        grid = np.zeros((args.grid_size, args.grid_size), dtype=np.int8)
        # Add random obstacles
        grid[np.random.rand(args.grid_size, args.grid_size) < args.obstacle_density] = 1
    
    # Invert grid if requested
    if args.invert_grid:
        print("Inverting grid values (0→1, 1→0)")
        grid = 1 - grid
    
    # Check if grid is too large and resize if needed
    original_size = grid.shape
    if grid.shape[0] > args.max_grid_size or grid.shape[1] > args.max_grid_size:
        print(f"Grid size {grid.shape} too large for visualization. Resizing to {args.max_grid_size}x{args.max_grid_size}...")
        grid = resize_grid(grid, args.max_grid_size)
        print(f"Grid resized from {original_size} to {grid.shape}")
    
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
    
    # Save the grid to the output directory
    grid_path = output_dir / "test_grid.npy"
    np.save(grid_path, grid)
    print(f"Grid saved to {grid_path}")
    
    # Compare different algorithms if requested
    if args.compare:
        compare_algorithms(
            grid=grid,
            start=start_pos,
            goal=goal_pos,
            output_dir=output_dir,
            diagonal=args.diagonal,
            timeout=args.timeout
        )
        return
    
    # Create navigation environment for visualization
    env = NavigationEnv(
        grid=grid,
        start_pos=start_pos,
        goal_pos=goal_pos,
        render_mode=None
    )
    
    # Run the appropriate algorithm
    if args.bidirectional:
        print("\nRunning bidirectional A* search...")
        path, metrics = bidirectional_a_star(
            grid=grid,
            start=start_pos,
            goal=goal_pos,
            heuristic=args.heuristic,
            diagonal=args.diagonal,
            timeout=args.timeout
        )
    else:
        print(f"\nRunning A* search with {args.heuristic} heuristic...")
        path, metrics = a_star_search(
            grid=grid,
            start=start_pos,
            goal=goal_pos,
            heuristic=args.heuristic,
            diagonal=args.diagonal,
            timeout=args.timeout,
            weight=args.weight
        )
    
    # Print results
    if path:
        print(f"Path found with length {len(path)}")
        print(f"Nodes explored: {metrics['nodes_explored']}")
        print(f"Time taken: {metrics['time_taken']:.4f} seconds")
        print(f"Path cost: {metrics['path_cost']:.1f}")
        
        # Visualize the path
        algorithm_name = "Bidirectional A*" if args.bidirectional else f"A* ({args.heuristic})"
        visualize_path(
            env=env,
            path=path,
            title=f"{algorithm_name} Path (Length: {len(path)})",
            save_path=output_dir / f"path_{args.heuristic}.png",
            show=True
        )
    else:
        print("No path found")


if __name__ == "__main__":
    main() 