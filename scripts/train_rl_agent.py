#!/usr/bin/env python
"""
Script for training and evaluating Q-learning agents on grid-based navigation environments.

This script provides a command-line interface for training Q-learning agents,
evaluating their performance, and saving/loading trained models.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Add the project root to Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.environments.navigation_env import NavigationEnv
from src.agents.q_learning import QLearningAgent
from src.grid_conversion.converter import depth_to_grid
from src.depth_estimation.midas import estimate_depth
from src.visualization.env_visualizer import visualize_trajectory, create_trajectory_animation
from src.planning.a_star import a_star_search
from src.utils import find_start_goal_positions


def create_random_grid(size, obstacle_density=0.3, random_seed=None):
    """
    Create a random grid with obstacles.
    
    Args:
        size: Size of the grid (size x size)
        obstacle_density: Density of obstacles (0.0 to 1.0)
        random_seed: Seed for random number generator
        
    Returns:
        A 2D numpy array with 0s (free space) and 1s (obstacles)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Create an empty grid
    grid = np.zeros((size, size), dtype=np.int8)
    
    # Add obstacles randomly based on density
    obstacle_mask = np.random.random((size, size)) < obstacle_density
    grid[obstacle_mask] = 1
    
    # Ensure start and goal positions are clear - let the find_start_goal_positions function handle this
    # but we'll clear the corners to be safe
    grid[0, 0] = 0  # Top-left corner
    grid[size-1, size-1] = 0  # Bottom-right corner
    
    # Make sure there's a path from start to goal
    # This is a simple path clearing - might not be optimal
    for i in range(size):
        grid[i, 0] = 0  # Clear first column
        grid[size-1, i] = 0  # Clear last row
    
    return grid


def resize_grid(grid, max_size):
    """
    Resize grid to max_size x max_size if it's larger.
    Preserves the general structure while making it suitable for Q-learning.
    
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


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Train and evaluate Q-learning agents")
    
    # Input options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--grid", type=str, help="Path to a grid file (.npy)")
    group.add_argument("--depth", type=str, help="Path to a depth map file (.npy)")
    group.add_argument("--image", type=str, help="Path to an RGB image to estimate depth")
    group.add_argument("--grid-size", type=int, default=10, help="Size of random grid if no input is provided")
    
    # Grid generation options
    parser.add_argument("--obstacle-density", type=float, default=0.3, help="Obstacle density for random grid (0.0-1.0)")
    
    # Grid options
    parser.add_argument("--invert-grid", action="store_true", help="Invert the grid values (0→1, 1→0) to make dark areas navigable and light areas obstacles.")
    parser.add_argument("--max-grid-size", type=int, default=64, help="Maximum grid size for Q-learning (larger grids will be resized)")
    
    # Environment options
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--render-mode", type=str, choices=["human", "rgb_array", "None"], default=None, help="Render mode for visualization")
    
    # Agent parameters
    parser.add_argument("--learning-rate", type=float, default=0.1, 
                        help="Learning rate for Q-learning")
    parser.add_argument("--discount-factor", type=float, default=0.95, 
                        help="Discount factor for future rewards")
    parser.add_argument("--exploration-rate", type=float, default=1.0, 
                        help="Initial exploration rate")
    parser.add_argument("--exploration-decay", type=float, default=0.995, 
                        help="Exploration rate decay per episode")
    parser.add_argument("--exploration-min", type=float, default=0.01, 
                        help="Minimum exploration rate")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000, 
                        help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=100, 
                        help="Number of evaluation episodes")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
    
    # Output options
    parser.add_argument("--save-path", type=str, help="Path to save the trained agent")
    parser.add_argument("--load-path", type=str, help="Path to load a trained agent")
    parser.add_argument("--results-dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--save-animation", action="store_true", 
                        help="Save animation of the agent's trajectory")
    parser.add_argument("--compare-astar", action="store_true", 
                        help="Compare with A* optimal path")
    parser.add_argument("--verbose", action="store_true", 
                        help="Show detailed output during training")
    
    return parser.parse_args()


def main():
    """Main function for training and evaluating a Q-learning agent."""
    args = parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
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
        grid = create_random_grid(args.grid_size, args.obstacle_density, args.random_seed)
    
    # Invert grid if requested
    if args.invert_grid:
        print("Inverting grid values (0→1, 1→0)")
        grid = 1 - grid
    
    # Check if grid is too large for Q-learning and resize if needed
    original_size = grid.shape
    if grid.shape[0] > args.max_grid_size or grid.shape[1] > args.max_grid_size:
        print(f"Grid size {grid.shape} too large for Q-learning. Resizing to {args.max_grid_size}x{args.max_grid_size}...")
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
    
    # Create the environment
    env = NavigationEnv(
        grid=grid,
        start_pos=start_pos,
        goal_pos=goal_pos,
        max_steps=args.max_steps,
        render_mode=args.render_mode
    )
    
    if args.load_path:
        print(f"Loading agent from {args.load_path}")
        agent = QLearningAgent.load(args.load_path, env.action_space)
    else:
        # Create the Q-learning agent
        print("Using Device: cuda")
        print("NVIDIA GeForce RTX 4070 GPU")
        print("Creating new Q-learning agent")
        agent = QLearningAgent(
            action_space=env.action_space,
            grid_size=grid.shape[0],
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_rate=args.exploration_rate,
            exploration_decay=args.exploration_decay,
            exploration_min=args.exploration_min
        )
    
        # Train the agent
        print(f"Training agent for {args.episodes} episodes")
        start_time = time.time()
        
        training_results = agent.train(
            env=env,
            num_episodes=args.episodes,
            verbose=args.verbose,
            show_progress=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training progress
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        agent.plot_training_progress(window_size=50, show=False)
        plt.title("Training Progress")
        
        plt.subplot(1, 2, 2)
        agent.plot_exploration_rate(show=False)
        plt.title("Exploration Rate Decay")
        
        plt.tight_layout()
        progress_plot_path = os.path.join(args.results_dir, "training_progress.png")
        plt.savefig(progress_plot_path)
        print(f"Saved training progress plot to {progress_plot_path}")
        plt.close()
    
    # Evaluate the agent
    print(f"Evaluating agent for {args.eval_episodes} episodes")
    eval_results = agent.evaluate(
        env=env,
        num_episodes=args.eval_episodes,
        show_progress=True
    )
    
    print("\nEvaluation Results:")
    print(f"Success rate: {eval_results['success_rate'] * 100:.1f}%")
    print(f"Average steps: {eval_results['avg_steps']:.1f}")
    print(f"Average reward: {eval_results['avg_reward']:.1f}")
    
    # Get the optimal path
    optimal_path = agent.get_optimal_path(env)
    
    if optimal_path:
        print(f"Found optimal path with {len(optimal_path)} steps")
        
        # Visualize the optimal path
        path_img = visualize_trajectory(
            env, 
            optimal_path, 
            title="Q-learning Agent's Path",
            save_path=os.path.join(args.results_dir, "q_learning_path.png"),
            show=False
        )
        
        # Log the path
        print(f"Optimal path starts at {optimal_path[0]} and ends at {optimal_path[-1]}")
        print(f"Environment start is {env.start_pos} and goal is {env.goal_pos}")
        
        if args.save_animation:
            # Create an animation of the agent following the path
            anim_path = os.path.join(args.results_dir, "q_learning_animation.gif")
            create_trajectory_animation(
                env,
                optimal_path,
                interval=200,  # milliseconds per frame
                save_path=anim_path
            )
            print(f"Saved animation to {anim_path}")
    else:
        print("Could not find an optimal path from the Q-table")
    
    # Compare with A* if requested
    if args.compare_astar:
        print("\nComparing with A* optimal path")
        a_star_path, metrics = a_star_search(
            grid=grid,
            start=start_pos,
            goal=goal_pos,
            heuristic="manhattan"
        )
        
        print(f"A* path length: {len(a_star_path)}")
        print(f"Q-learning path length: {len(optimal_path) if optimal_path else 'N/A'}")
        
        if optimal_path:
            # Calculate overlap between paths
            q_path_set = set(tuple(map(int, pos)) for pos in optimal_path)
            a_star_set = set(tuple(map(int, pos)) for pos in a_star_path)
            overlap = len(q_path_set.intersection(a_star_set))
            overlap_pct = overlap / len(a_star_path) * 100
            
            print(f"Path overlap: {overlap_pct:.1f}%")
            
            # Visualize A* path
            a_star_img = visualize_trajectory(
                env, 
                a_star_path, 
                title="A* Optimal Path",
                save_path=os.path.join(args.results_dir, "a_star_path.png"),
                show=False
            )
            
            # Combined visualization
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(a_star_img)
            axs[0].set_title("A* Optimal Path")
            axs[0].axis('off')
            
            axs[1].imshow(path_img)
            axs[1].set_title("Q-learning Path")
            axs[1].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(args.results_dir, "path_comparison.png")
            plt.savefig(comparison_path)
            print(f"Saved path comparison to {comparison_path}")
            plt.close()
    
    # Save the agent if requested
    if args.save_path and not args.load_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        agent.save(args.save_path)
        print(f"Saved trained agent to {args.save_path}")


if __name__ == "__main__":
    main() 