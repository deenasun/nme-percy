#!/usr/bin/env python
"""Script to test the navigation environment.

This script demonstrates how to create and interact with the navigation
environment using a grid generated from a depth map.
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

from src.environments.navigation_env import NavigationEnv
from src.visualization.env_visualizer import (
    visualize_path, visualize_trajectory, 
    create_trajectory_animation, display_env_info
)
from src.grid_conversion.converter import depth_to_grid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Test the navigation environment.")
    
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
        "--grid-size",
        type=int,
        default=32,
        help="Size of the grid if generating from depth map. Defaults to 32."
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Threshold for obstacle detection if generating from depth map. Defaults to 0.6."
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
        "--steps",
        type=int,
        default=100,
        help="Number of steps to run in the environment. Defaults to 100."
    )
    
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["human", "rgb_array", "none"],
        default="human",
        help="Render mode for the environment. Defaults to 'human'."
    )
    
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save an animation of the agent's trajectory."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/navigation",
        help="Directory to save output files. Defaults to 'results/navigation'."
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


def load_or_generate_grid(args: argparse.Namespace) -> np.ndarray:
    """Load a grid from a file or generate one from a depth map.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        A 2D numpy array representing the grid environment.
        
    Raises:
        ValueError: If neither grid nor depth map path is provided.
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
    else:
        # Generate a simple test grid if no input is provided
        print("No grid or depth map provided. Generating a simple test grid.")
        grid_size = args.grid_size
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        
        # Add walls around the edges
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        # Add some random obstacles
        np.random.seed(42)  # For reproducibility
        num_obstacles = grid_size * grid_size // 5  # 20% of cells are obstacles
        for _ in range(num_obstacles):
            row = np.random.randint(1, grid_size - 1)
            col = np.random.randint(1, grid_size - 1)
            grid[row, col] = 1
        
        # Ensure start and goal positions are free
        start_pos = parse_position(args.start_pos)
        if 0 <= start_pos[0] < grid_size and 0 <= start_pos[1] < grid_size:
            grid[start_pos] = 0
        
        if args.goal_pos:
            goal_pos = parse_position(args.goal_pos)
            if 0 <= goal_pos[0] < grid_size and 0 <= goal_pos[1] < grid_size:
                grid[goal_pos] = 0
        else:
            # Clear the bottom-right corner for the default goal
            grid[grid_size - 2:, grid_size - 2:] = 0
    
    return grid


def main() -> None:
    """Main function to test the navigation environment."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load or generate grid
    grid = load_or_generate_grid(args)
    
    # Parse positions
    start_pos = parse_position(args.start_pos)
    goal_pos = parse_position(args.goal_pos) if args.goal_pos else None
    
    # Create environment
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = NavigationEnv(
        grid=grid,
        start_pos=start_pos,
        goal_pos=goal_pos,
        max_steps=args.steps,
        render_mode=render_mode
    )
    
    # Display environment information
    display_env_info(env)
    
    # Run a random agent in the environment
    print("\nRunning random agent...\n")
    
    # Reset environment
    observation, _ = env.reset()
    
    # Initialize variables
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    trajectory = [env.agent_pos]  # Start with initial position
    rewards = []
    
    # Main loop
    while not (done or truncated) and step_count < args.steps:
        # Take a random action
        action = env.action_space.sample()
        
        # Step in the environment
        observation, reward, done, truncated, info = env.step(action)
        
        # Update tracking variables
        total_reward += reward
        step_count += 1
        trajectory.append(env.agent_pos)
        rewards.append(reward)
        
        # Print step information
        print(f"Step {step_count}: Action={action}, Position={env.agent_pos}, Reward={reward:.1f}, Done={done}, Truncated={truncated}")
    
    # Print final results
    print(f"\nSimulation complete after {step_count} steps")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Final position: {env.agent_pos}")
    
    if env.agent_pos == env.goal_pos:
        print("Goal reached!")
    
    # Visualize trajectory
    visualize_trajectory(
        env=env,
        trajectory=trajectory,
        rewards=rewards,
        title=f"Random Agent Trajectory (Reward: {total_reward:.1f})",
        save_path=output_dir / "random_trajectory.png"
    )
    
    # Create and save animation if requested
    if args.save_animation:
        print("\nCreating trajectory animation...")
        animation_path = output_dir / "random_trajectory.gif"
        create_trajectory_animation(
            env=env,
            trajectory=trajectory,
            title=f"Random Agent (Reward: {total_reward:.1f})",
            save_path=animation_path,
            show=False
        )
        print(f"Animation saved to {animation_path}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main() 