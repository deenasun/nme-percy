"""RL agent visualization utilities.

This module provides functions for visualizing Q-learning agent results,
including Q-values, learning progress, and policies.
"""

from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Arrow
import seaborn as sns
from matplotlib.animation import FuncAnimation

from src.environments.navigation_env import NavigationEnv


def visualize_q_values(
    q_table: np.ndarray,
    grid: np.ndarray,
    start_pos: Tuple[int, int] = (0, 0),
    goal_pos: Optional[Tuple[int, int]] = None,
    title: str = "Q-Values",
    action_labels: List[str] = ["Up", "Right", "Down", "Left"],
    fig_size: Tuple[int, int] = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Visualize the Q-values for each state and action.
    
    Args:
        q_table: The Q-table (state_row, state_col, action) to visualize.
        grid: The grid environment (0 for free space, 1 for obstacle).
        start_pos: The start position as (row, col).
        goal_pos: The goal position as (row, col). If None, uses bottom-right corner.
        title: Title for the plot.
        action_labels: Labels for each action.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the visualization. If None, won't save.
        show: Whether to display the plot.
        dpi: DPI for the saved figure.
    """
    if goal_pos is None:
        goal_pos = (grid.shape[0] - 1, grid.shape[1] - 1)
    
    n_actions = q_table.shape[2]
    grid_size = grid.shape[0]
    
    # Create a figure with subplots for each action
    fig, axs = plt.subplots(2, 2, figsize=fig_size)
    axs = axs.flatten()
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_red", ["blue", "white", "red"], N=100)
    
    # Find min and max Q-values for consistent colormap
    vmin = np.min(q_table)
    vmax = np.max(q_table)
    
    # Plot each action's Q-values
    for a in range(n_actions):
        ax = axs[a]
        
        # Create a masked array where obstacles are masked
        masked_q = np.ma.array(q_table[:, :, a], mask=grid == 1)
        
        # Plot Q-values
        im = ax.imshow(masked_q, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Plot obstacles
        obstacle_mask = grid == 1
        if np.any(obstacle_mask):
            # Create a 2D meshgrid of coordinates
            y, x = np.mgrid[:grid_size, :grid_size]
            
            # Use black color for obstacles
            ax.scatter(x[obstacle_mask], y[obstacle_mask], 
                      c='black', marker='s', s=50)
        
        # Mark start and goal
        ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, label="Start")
        ax.plot(goal_pos[1], goal_pos[0], 'rD', markersize=10, label="Goal")
        
        # Draw grid lines
        if grid_size <= 20:  # Only for smaller grids
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
            # Hide major tick labels
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Label the subplot with the action
        ax.set_title(f"Action: {action_labels[a]}")
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
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


def visualize_max_q_values(
    q_table: np.ndarray,
    grid: np.ndarray,
    start_pos: Tuple[int, int] = (0, 0),
    goal_pos: Optional[Tuple[int, int]] = None,
    title: str = "Maximum Q-Values",
    fig_size: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Visualize the maximum Q-value for each state.
    
    Args:
        q_table: The Q-table (state_row, state_col, action) to visualize.
        grid: The grid environment (0 for free space, 1 for obstacle).
        start_pos: The start position as (row, col).
        goal_pos: The goal position as (row, col). If None, uses bottom-right corner.
        title: Title for the plot.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the visualization. If None, won't save.
        show: Whether to display the plot.
        dpi: DPI for the saved figure.
    """
    if goal_pos is None:
        goal_pos = (grid.shape[0] - 1, grid.shape[1] - 1)
    
    grid_size = grid.shape[0]
    
    # Get max Q-value for each state
    max_q_values = np.max(q_table, axis=2)
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Create masked array where obstacles are masked
    masked_max_q = np.ma.array(max_q_values, mask=grid == 1)
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_red", ["blue", "white", "red"], N=100)
    
    # Plot max Q-values
    im = plt.imshow(masked_max_q, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Max Q-Value')
    
    # Mark start and goal
    plt.plot(start_pos[1], start_pos[0], 'go', markersize=10, label="Start")
    plt.plot(goal_pos[1], goal_pos[0], 'rD', markersize=10, label="Goal")
    
    # Draw grid lines
    if grid_size <= 20:
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, grid_size, 1), [])
        plt.yticks(np.arange(-0.5, grid_size, 1), [])
    
    # Add title
    plt.title(title)
    
    # Add legend
    plt.legend(loc="upper right")
    
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


def visualize_policy(
    q_table: np.ndarray,
    grid: np.ndarray,
    start_pos: Tuple[int, int] = (0, 0),
    goal_pos: Optional[Tuple[int, int]] = None,
    title: str = "Learned Policy",
    action_symbols: List[str] = ["↑", "→", "↓", "←"],
    fig_size: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Visualize the learned policy (best action) for each state.
    
    Args:
        q_table: The Q-table (state_row, state_col, action) to visualize.
        grid: The grid environment (0 for free space, 1 for obstacle).
        start_pos: The start position as (row, col).
        goal_pos: The goal position as (row, col). If None, uses bottom-right corner.
        title: Title for the plot.
        action_symbols: Symbols for each action.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the visualization. If None, won't save.
        show: Whether to display the plot.
        dpi: DPI for the saved figure.
    """
    if goal_pos is None:
        goal_pos = (grid.shape[0] - 1, grid.shape[1] - 1)
    
    grid_size = grid.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot grid (grayscale with obstacles as black)
    ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1)
    
    # Define arrow properties based on grid size
    if grid_size <= 10:
        arrow_size = 0.4
        text_size = 12
    elif grid_size <= 20:
        arrow_size = 0.3
        text_size = 10
    else:
        arrow_size = 0.2
        text_size = 8
    
    # Plot policy arrows
    for row in range(grid_size):
        for col in range(grid_size):
            # Skip obstacles
            if grid[row, col] == 1:
                continue
                
            # Skip goal position
            if (row, col) == goal_pos:
                continue
                
            # Get best action
            best_action = np.argmax(q_table[row, col])
            
            # Draw arrow or symbol for best action
            if grid_size <= 25:  # Use arrows for smaller grids
                # Define arrow direction
                if best_action == 0:  # Up
                    dx, dy = 0, -arrow_size
                elif best_action == 1:  # Right
                    dx, dy = arrow_size, 0
                elif best_action == 2:  # Down
                    dx, dy = 0, arrow_size
                else:  # Left
                    dx, dy = -arrow_size, 0
                
                # Create arrow
                arrow = Arrow(col, row, dx, dy, width=0.3, 
                             color='blue', alpha=0.7)
                ax.add_patch(arrow)
            else:  # Use symbols for larger grids
                ax.text(col, row, action_symbols[best_action], 
                       fontsize=text_size, ha='center', va='center', color='blue')
    
    # Mark start and goal
    ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, label="Start")
    ax.plot(goal_pos[1], goal_pos[0], 'rD', markersize=10, label="Goal")
    
    # Grid lines for smaller grids
    if grid_size <= 20:
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add title
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc="upper right")
    
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


def plot_training_progress(
    episode_rewards: List[float],
    episode_steps: List[int],
    success_rate: Optional[List[float]] = None,
    window_size: int = 50,
    title: str = "Q-Learning Training Progress",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Plot the training progress of a Q-learning agent.
    
    Args:
        episode_rewards: List of rewards per episode.
        episode_steps: List of steps per episode.
        success_rate: Optional list of success rates.
        window_size: Window size for rolling average.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
        dpi: DPI for saved figure.
    """
    # Calculate moving averages
    rewards_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    steps_avg = np.convolve(episode_steps, np.ones(window_size)/window_size, mode='valid')
    
    # Create figure with multiple plots
    fig, axes = plt.subplots(3 if success_rate else 2, 1, figsize=figsize, sharex=True)
    
    # Plot episode rewards
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(np.arange(window_size-1, len(episode_rewards)), 
            rewards_avg, color='blue', linewidth=2, 
            label=f'Moving Avg (window={window_size})')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot episode steps
    ax2 = axes[1]
    ax2.plot(episode_steps, alpha=0.3, color='green', label='Episode Steps')
    ax2.plot(np.arange(window_size-1, len(episode_steps)), 
            steps_avg, color='green', linewidth=2, 
            label=f'Moving Avg (window={window_size})')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Steps')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot success rate if provided
    if success_rate:
        ax3 = axes[2]
        ax3.plot(success_rate, color='red', linewidth=2, label='Success Rate')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Training Success Rate')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # Set common x-label
    axes[-1].set_xlabel('Episode')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if path provided
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


def plot_exploration_rate(
    exploration_rates: List[float],
    title: str = "Exploration Rate Decay",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Plot the exploration rate decay during training.
    
    Args:
        exploration_rates: List of exploration rates per episode.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
        dpi: DPI for saved figure.
    """
    plt.figure(figsize=figsize)
    
    plt.plot(exploration_rates, color='purple', linewidth=2)
    plt.ylabel('Exploration Rate (ε)')
    plt.xlabel('Episode')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for start and end values
    plt.annotate(f'Start: {exploration_rates[0]:.2f}', 
                xy=(0, exploration_rates[0]),
                xytext=(10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f'End: {exploration_rates[-1]:.2f}', 
                xy=(len(exploration_rates)-1, exploration_rates[-1]),
                xytext=(-10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'))
    
    # Save if path provided
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


def create_q_learning_animation(
    env: NavigationEnv,
    q_tables: List[np.ndarray],
    episode_indices: List[int],
    title: str = "Q-Learning Progress",
    interval: int = 500,  # milliseconds per frame
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[FuncAnimation]:
    """Create an animation showing how Q-values evolve during training.
    
    Args:
        env: The navigation environment.
        q_tables: List of Q-tables at different training stages.
        episode_indices: Episode numbers corresponding to each Q-table.
        title: Title for the animation.
        interval: Time between frames in milliseconds.
        save_path: Path to save the animation. If None, won't save.
        show: Whether to display the animation.
        
    Returns:
        The animation object if show is True, None otherwise.
    """
    if not q_tables:
        print("Error: No Q-tables provided")
        return None
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get max Q-value across all tables for consistent colormap
    vmax = max([np.max(q) for q in q_tables])
    vmin = min([np.min(q) for q in q_tables])
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_red", ["blue", "white", "red"], N=100)
    
    # Initialize plot with first Q-table's max values
    max_q_values = np.max(q_tables[0], axis=2)
    masked_max_q = np.ma.array(max_q_values, mask=env.grid == 1)
    
    # Create initial plot
    im = ax.imshow(masked_max_q, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Q-Value')
    
    # Mark start and goal
    ax.plot(env.start_pos[1], env.start_pos[0], 'go', markersize=10, label="Start")
    ax.plot(env.goal_pos[1], env.goal_pos[0], 'rD', markersize=10, label="Goal")
    
    # Add grid lines if the grid is small enough
    if env.grid_size <= 20:
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add title with episode counter
    title_obj = ax.set_title(f"{title} - Episode 0")
    
    # Add legend
    ax.legend(loc="upper right")
    
    # Animation update function
    def update(frame):
        # Update max Q-values
        max_q_values = np.max(q_tables[frame], axis=2)
        masked_max_q = np.ma.array(max_q_values, mask=env.grid == 1)
        
        # Update image
        im.set_array(masked_max_q)
        
        # Update title with episode number
        title_obj.set_text(f"{title} - Episode {episode_indices[frame]}")
        
        return [im, title_obj]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(q_tables),
                         blit=True, interval=interval)
    
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


def compare_paths(
    env: NavigationEnv,
    q_learning_path: List[Tuple[int, int]],
    reference_path: List[Tuple[int, int]],
    q_learning_label: str = "Q-Learning Path",
    reference_label: str = "A* Path",
    title: str = "Path Comparison",
    fig_size: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    dpi: int = 100
) -> None:
    """Compare the Q-learning path with a reference path (e.g., A* path).
    
    Args:
        env: The navigation environment.
        q_learning_path: The path from Q-learning.
        reference_path: The reference path (e.g., from A*).
        q_learning_label: Label for the Q-learning path.
        reference_label: Label for the reference path.
        title: Title for the plot.
        fig_size: Size of the figure (width, height) in inches.
        save_path: Path to save the visualization. If None, won't save.
        show: Whether to display the plot.
        dpi: DPI for the saved figure.
    """
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Plot grid in grayscale
    plt.imshow(env.grid, cmap='gray_r', vmin=0, vmax=1)
    
    # Plot paths
    if q_learning_path and len(q_learning_path) > 1:
        rows, cols = zip(*q_learning_path)
        plt.plot(cols, rows, 'b-', linewidth=3, alpha=0.6, label=q_learning_label)
        plt.plot(cols, rows, 'bo', markersize=4)
    
    if reference_path and len(reference_path) > 1:
        rows, cols = zip(*reference_path)
        plt.plot(cols, rows, 'r--', linewidth=3, alpha=0.6, label=reference_label)
        plt.plot(cols, rows, 'ro', markersize=4)
    
    # Mark start and goal positions
    plt.plot(env.start_pos[1], env.start_pos[0], 'go', markersize=10, label="Start")
    plt.plot(env.goal_pos[1], env.goal_pos[0], 'rD', markersize=10, label="Goal")
    
    # Add grid lines if the grid is small enough
    if env.grid_size <= 30:
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, env.grid_size, 1), [])
        plt.yticks(np.arange(-0.5, env.grid_size, 1), [])
    
    # Add legend
    plt.legend(loc="best")
    
    # Add title
    plt.title(title)
    
    # Calculate and display path statistics
    q_len = len(q_learning_path) if q_learning_path else 0
    ref_len = len(reference_path) if reference_path else 0
    
    if q_learning_path and reference_path:
        # Calculate overlap
        q_set = set(tuple(pos) for pos in q_learning_path)
        ref_set = set(tuple(pos) for pos in reference_path)
        overlap = len(q_set.intersection(ref_set))
        overlap_pct = overlap / len(ref_set) * 100
        
        # Display statistics
        plt.figtext(0.5, 0.01, 
                   f"{q_learning_label} length: {q_len} | "
                   f"{reference_label} length: {ref_len} | "
                   f"Overlap: {overlap_pct:.1f}%", 
                   ha="center", fontsize=10,
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
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