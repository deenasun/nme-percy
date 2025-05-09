"""Environment visualization utilities.

This module provides functions for visualizing the navigation environment,
including rendering agent trajectories and paths through the grid.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import gymnasium as gym
import logging

from src.environments.navigation_env import NavigationEnv
from src.grid_conversion.converter import FREE_SPACE, OBSTACLE, START_POINT, GOAL_POINT

# Set up logging
logger = logging.getLogger(__name__)

def visualize_path(
    env: NavigationEnv,
    path: List[Tuple[int, int]],
    explored_nodes: Optional[List[Tuple[int, int]]] = None,
    title: str = "Path Visualization",
    fig_size: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Visualize a path on a grid.
    
    Args:
        env: Navigation environment
        path: List of (row, col) positions representing the path
        explored_nodes: List of (row, col) positions that were explored during search
        title: Title for the plot
        fig_size: Size of the figure
        save_path: Path to save the figure (None = don't save)
        show: Whether to show the figure
    """
    # Create a copy of the grid for visualization
    vis_grid = env.grid.copy()
    
    # Mark start and goal
    if hasattr(env, 'agent_pos'):
        start_pos = env.agent_pos
    else:
        # Fallback to the first position in the path
        start_pos = path[0] if path else (0, 0)
        
    goal_pos = env.goal_pos
    
    vis_grid[start_pos] = START_POINT
    vis_grid[goal_pos] = GOAL_POINT
    
    # Create a colored grid for visualization
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red', 'blue', 'yellow'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=fig_size)
    plt.imshow(vis_grid, cmap=cmap, norm=norm)
    
    # Plot explored nodes if provided
    if explored_nodes:
        explored_y, explored_x = zip(*[(pos[0], pos[1]) for pos in explored_nodes 
                                      if not (np.array_equal(pos, start_pos) or np.array_equal(pos, goal_pos))])
        plt.scatter(explored_x, explored_y, c='pink', marker='s', 
                   s=50, alpha=0.5, label='Explored')
    
    # Plot the path
    if path:
        path_y, path_x = zip(*[(pos[0], pos[1]) for pos in path 
                              if not (np.array_equal(pos, start_pos) or np.array_equal(pos, goal_pos))])
        plt.scatter(path_x, path_y, c='blue', marker='o', 
                   s=50, alpha=0.7, label='Path')
        
        # Add direction arrows
        for i in range(len(path)-1):
            plt.arrow(
                path[i][1], path[i][0],
                path[i+1][1] - path[i][1], path[i+1][0] - path[i][0],
                color='blue', head_width=0.3, head_length=0.3, length_includes_head=True
            )
    
    # Add labels and legend
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
               ncol=3, frameon=True)
    plt.grid(False)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved path visualization to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_trajectory(
    env: NavigationEnv,
    trajectory: List[Tuple[int, int]],
    title: str = "Agent Trajectory",
    fig_size: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Visualize an agent's trajectory on a grid.
    
    Args:
        env: Navigation environment
        trajectory: List of (row, col) positions representing the trajectory
        title: Title for the plot
        fig_size: Size of the figure
        save_path: Path to save the figure (None = don't save)
        show: Whether to show the figure
    """
    # Create a copy of the grid for visualization
    vis_grid = env.grid.copy()
    
    # Mark start and goal
    if hasattr(env, 'agent_pos'):
        start_pos = env.agent_pos
    else:
        # Fallback to the first position in the trajectory
        start_pos = trajectory[0] if trajectory else (0, 0)
        
    goal_pos = env.goal_pos
    
    vis_grid[start_pos] = START_POINT
    vis_grid[goal_pos] = GOAL_POINT
    
    # Create a colored grid for visualization
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=fig_size)
    plt.imshow(vis_grid, cmap=cmap, norm=norm)
    
    # Plot the trajectory
    if trajectory:
        # Filter out start and goal positions
        filtered_trajectory = [pos for pos in trajectory 
                               if not (np.array_equal(pos, start_pos) or np.array_equal(pos, goal_pos))]
        
        if filtered_trajectory:
            traj_y, traj_x = zip(*filtered_trajectory)
            plt.scatter(traj_x, traj_y, c='blue', marker='o', 
                       s=50, alpha=0.7, label='Trajectory')
            
            # Add direction arrows
            for i in range(len(trajectory)-1):
                plt.arrow(
                    trajectory[i][1], trajectory[i][0],
                    trajectory[i+1][1] - trajectory[i][1], 
                    trajectory[i+1][0] - trajectory[i][0],
                    color='blue', head_width=0.3, head_length=0.3, 
                    length_includes_head=True
                )
    
    # Add labels and legend
    plt.title(title)
    plt.grid(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
               ncol=3, frameon=True)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved trajectory visualization to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def create_trajectory_animation(
    env: NavigationEnv,
    trajectory: List[Tuple[int, int]],
    title: str = "Agent Navigation",
    interval: int = 200,  # milliseconds between frames
    fig_size: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[FuncAnimation]:
    """Create an animation of an agent following a trajectory.
    
    Args:
        env: The navigation environment.
        trajectory: List of (row, col) coordinates defining the agent's trajectory.
        title: Title for the animation.
        interval: Time between frames in milliseconds.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the animation. If None, won't save.
        show: Whether to display the animation.
        
    Returns:
        The animation object if show is True, None otherwise.
    """
    if not trajectory:
        print("Error: Empty trajectory")
        return None
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Prepare grid for visualization
    vis_grid = env.grid.copy()
    
    # Plot grid
    cax = ax.imshow(vis_grid, cmap='gray_r', vmin=0, vmax=1)
    
    # Mark start and goal positions
    if hasattr(env, 'agent_pos'):
        start_pos = env.agent_pos
    else:
        # Fallback to the first position in the trajectory
        start_pos = trajectory[0] if trajectory else (0, 0)
    
    start_marker = plt.Rectangle((start_pos[1] - 0.5, start_pos[0] - 0.5), 1, 1, 
                                 fill=True, color='green', alpha=0.7)
    goal_marker = plt.Rectangle((env.goal_pos[1] - 0.5, env.goal_pos[0] - 0.5), 1, 1, 
                                fill=True, color='red', alpha=0.7)
    ax.add_patch(start_marker)
    ax.add_patch(goal_marker)
    
    # Add grid lines
    if env.grid_size <= 50:
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Initialize agent marker (will be updated in animation)
    agent_pos = trajectory[0]
    agent_marker = plt.Rectangle((agent_pos[1] - 0.5, agent_pos[0] - 0.5), 1, 1, 
                                fill=True, color='blue', alpha=0.7)
    ax.add_patch(agent_marker)
    
    # Track the path (will grow over time)
    path_line, = ax.plot([], [], 'b-', linewidth=2)
    
    # Add title
    ax.set_title(title)
    
    # Animation initialization function
    def init():
        path_line.set_data([], [])
        return path_line, agent_marker
    
    # Animation update function
    def update(frame):
        pos = trajectory[frame]
        
        # Update agent position
        agent_marker.set_xy((pos[1] - 0.5, pos[0] - 0.5))
        
        # Update path line
        path_points = trajectory[:frame+1]
        if path_points:
            rows, cols = zip(*path_points)
            path_line.set_data(cols, rows)
        
        return path_line, agent_marker
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(trajectory),
                          init_func=init, blit=True, interval=interval)
    
    # Save animation if requested
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save as GIF or MP4
        if str(save_path).endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        else:
            # Default to MP4
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
    
    # Show animation
    if show:
        plt.show()
        return anim
    else:
        plt.close()
        return None


def interactive_simulation(env: NavigationEnv, policy_fn=None, max_steps: int = 100) -> Tuple[List[Tuple[int, int]], float]:
    """Run an interactive simulation of the environment with a policy function.
    
    Args:
        env: The navigation environment.
        policy_fn: Function that takes an observation and returns an action.
                   If None, uses a random policy.
        max_steps: Maximum number of steps to execute.
        
    Returns:
        Tuple containing the trajectory and total reward.
    """
    if policy_fn is None:
        # Default to random policy
        policy_fn = lambda obs: env.action_space.sample()
    
    # Reset environment
    obs, _ = env.reset()
    
    # Initialize variables
    done = False
    truncated = False
    total_reward = 0
    trajectory = [env.agent_pos]  # Start with initial position
    
    # Main loop
    step = 0
    while not (done or truncated) and step < max_steps:
        # Get action from policy
        action = policy_fn(obs)
        
        # Take step in environment
        obs, reward, done, truncated, _ = env.step(action)
        
        # Update tracking variables
        total_reward += reward
        trajectory.append(env.agent_pos)
        step += 1
    
    return trajectory, total_reward


def display_env_info(env: NavigationEnv) -> None:
    """Display information about the navigation environment.
    
    Args:
        env: The navigation environment to display information about.
    """
    # Calculate statistics
    obstacle_count = np.sum(env.grid == OBSTACLE)
    free_count = np.sum(env.grid == FREE_SPACE)
    total_cells = env.grid_size * env.grid_size
    
    # Print environment information
    print("Navigation Environment Information")
    print("==================================")
    print(f"Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"Agent Position: {env.agent_pos}")
    print(f"Goal Position: {env.goal_pos}")
    print(f"Manhattan Distance (Agent to Goal): {abs(env.agent_pos[0] - env.goal_pos[0]) + abs(env.agent_pos[1] - env.goal_pos[1])}")
    print(f"Obstacles: {obstacle_count} cells ({obstacle_count / total_cells:.1%})")
    print(f"Free Space: {free_count} cells ({free_count / total_cells:.1%})")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Maximum Steps: {env.max_steps}") 