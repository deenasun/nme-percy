"""Navigation environment for grid-based reinforcement learning.

This module implements a custom Gymnasium environment that represents
a navigable grid environment derived from depth maps. The environment
enables an agent to navigate from a start position to a goal position,
avoiding obstacles along the way.
"""

from typing import Dict, Tuple, Optional, Union, Any, List, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

class NavigationEnv(gym.Env):
    """A custom Gymnasium environment for grid-based navigation.
    
    This environment takes a grid representation (derived from a depth map),
    and allows an agent to navigate from a start position to a goal position.
    The grid contains obstacles (1) and free spaces (0).
    
    Attributes:
        grid (np.ndarray): Binary grid where 0 is free space and 1 is obstacle.
        grid_size (int): Size of the grid (assumed to be square).
        goal_pos (Tuple[int, int]): Position of the goal.
        start_pos (Tuple[int, int]): Starting position of the agent.
        agent_pos (Tuple[int, int]): Current position of the agent.
        action_space (spaces.Discrete): The action space (4 actions: up, right, down, left).
        observation_space (spaces.Dict): The observation space.
        max_steps (int): Maximum number of steps before episode termination.
        step_count (int): Counter for steps taken in the current episode.
        reward_fn (Callable): Custom reward function.
        render_mode (str): The mode for rendering ('human', 'rgb_array', or None).
        window_size (int): Size of the rendering window.
        window (Optional[Any]): Window object for rendering.
        clock (Optional[Any]): Clock object for rendering.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid: np.ndarray,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        max_steps: int = 100,
        reward_fn: Optional[Callable] = None,
        render_mode: Optional[str] = None,
        window_size: int = 512
    ) -> None:
        """Initialize the navigation environment.
        
        Args:
            grid: Binary grid where 0 is free space and 1 is obstacle.
            start_pos: Starting position of the agent (row, column). If None, defaults to top-left.
            goal_pos: Position of the goal (row, column). If None, defaults to bottom-right.
            max_steps: Maximum number of steps before episode termination.
            reward_fn: Custom reward function. If None, uses the default reward.
            render_mode: The mode for rendering ('human', 'rgb_array', or None).
            window_size: Size of the rendering window.
            
        Raises:
            ValueError: If the grid is not 2D, if start_pos or goal_pos are not
                        valid positions, or if they coincide with obstacles.
        """
        super().__init__()
        
        # Validate grid
        if grid.ndim != 2:
            raise ValueError(f"Grid must be 2D, got {grid.ndim}D")
        
        # Store parameters
        self.grid = grid.copy()  # Copy to avoid external modifications
        self.grid_size = grid.shape[0]  # Assuming square grid
        
        # Set default start and goal positions if not provided
        if start_pos is None:
            # Default to top-left corner
            start_pos = (0, 0)
            # Ensure it's not an obstacle
            if self.grid[start_pos] == 1:
                logger.warning("Default start position is an obstacle. Finding the first non-obstacle cell.")
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if self.grid[i, j] == 0:
                            start_pos = (i, j)
                            break
                    if self.grid[start_pos] == 0:
                        break
        
        if goal_pos is None:
            # Default to bottom-right corner
            goal_pos = (self.grid_size - 1, self.grid_size - 1)
            # Ensure it's not an obstacle
            if self.grid[goal_pos] == 1:
                logger.warning("Default goal position is an obstacle. Finding the last non-obstacle cell.")
                for i in range(self.grid_size - 1, -1, -1):
                    for j in range(self.grid_size - 1, -1, -1):
                        if self.grid[i, j] == 0:
                            goal_pos = (i, j)
                            break
                    if self.grid[goal_pos] == 0:
                        break
        
        # Validate positions
        for pos, name in [(start_pos, "start_pos"), (goal_pos, "goal_pos")]:
            if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
                raise ValueError(f"Invalid {name}: {pos}. Must be within grid bounds.")
            if self.grid[pos] == 1:
                raise ValueError(f"{name} {pos} coincides with an obstacle.")
        
        # Initialize agent position
        self.agent_pos = start_pos
        self.goal_pos = goal_pos
        # Store initial start position for resets
        self.start_pos = start_pos
        
        # Define action space: up (0), right (1), down (2), left (3)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'agent_position': spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            'goal_position': spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            'local_grid': spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8)
        })
        
        # Step limit
        self.max_steps = max_steps
        self.step_count = 0
        
        # Custom reward function
        self.reward_fn = reward_fn
        
        # Rendering setup
        self.render_mode = render_mode
        self.window_size = window_size
        self.window = None
        self.clock = None
        
        # For pygame rendering
        if self.render_mode == "human":
            self._init_rendering()
        
        logger.info(f"Environment created with grid size {self.grid_size}x{self.grid_size}")
        logger.info(f"Start position: {self.agent_pos}")
        logger.info(f"Goal position: {goal_pos}")
        logger.info(f"Max steps: {self.max_steps}")
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation.
        
        Returns:
            A dictionary containing the grid, agent position, and goal position.
        """
        return {
            'agent_position': np.array(self.agent_pos, dtype=np.int32),
            'goal_position': np.array(self.goal_pos, dtype=np.int32),
            'local_grid': self._get_local_grid()
        }
    
    def _get_local_grid(self, window_size: int = 5) -> np.ndarray:
        """Get a local view of the grid around the agent.
        
        Args:
            window_size: Size of the window (window_size x window_size)
            
        Returns:
            A numpy array representing the local view
        """
        # Initialize with all obstacles (for padding)
        local_view = np.ones((window_size, window_size), dtype=np.int8)
        
        # Calculate window boundaries
        half_size = window_size // 2
        row_start = max(0, self.agent_pos[0] - half_size)
        row_end = min(self.grid_size, self.agent_pos[0] + half_size + 1)
        col_start = max(0, self.agent_pos[1] - half_size)
        col_end = min(self.grid_size, self.agent_pos[1] + half_size + 1)
        
        # Calculate corresponding indices in the local view
        local_row_start = max(0, half_size - self.agent_pos[0])
        local_col_start = max(0, half_size - self.agent_pos[1])
        
        # Fill in the local view with grid values
        grid_height = row_end - row_start
        grid_width = col_end - col_start
        local_view[
            local_row_start:local_row_start+grid_height,
            local_col_start:local_col_start+grid_width
        ] = self.grid[row_start:row_end, col_start:col_end]
        
        return local_view
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state.
        
        Returns:
            A dictionary containing information such as distance to goal.
        """
        # Manhattan distance to goal
        manhattan_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        
        return {
            "distance_to_goal": manhattan_distance,
            "steps": self.step_count
        }
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to the initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset. Can include custom start_pos.
            
        Returns:
            Tuple containing the initial observation and info dictionary.
        """
        # Initialize RNG
        super().reset(seed=seed)
        
        # Reset agent position to initial start position or custom position from options
        if options and 'start_pos' in options:
            pos = options['start_pos']
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size and self.grid[pos] == 0:
                self.agent_pos = pos
        else:
            # Reset to the saved initial start position
            self.agent_pos = self.start_pos
        
        # Reset step counter
        self.step_count = 0
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Store previous position for reward calculation
        prev_position = self.agent_pos
        prev_distance = abs(prev_position[0] - self.goal_pos[0]) + abs(prev_position[1] - self.goal_pos[1])
        
        # Increment step counter
        self.step_count += 1
        
        # Move agent
        row, col = self.agent_pos
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.grid_size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.grid_size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
        
        # Check if new position is valid (not an obstacle)
        new_position = (row, col)
        hit_obstacle = False
        
        if self.grid[new_position] == 1:  # Obstacle
            # Stay in current position if obstacle
            new_position = self.agent_pos
            hit_obstacle = True
        
        # Update agent position
        self.agent_pos = new_position
        
        # Calculate current distance to goal
        current_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        
        # Check if goal reached
        goal_reached = np.array_equal(self.agent_pos, self.goal_pos)
        
        # Check if max steps reached
        timeout = self.step_count >= self.max_steps
        
        # Calculate reward
        if self.reward_fn:
            # Custom reward function
            reward = self.reward_fn(self.grid, self.agent_pos, self.goal_pos, goal_reached, self.step_count)
        else:
            # Default reward function
            if goal_reached:
                reward = 100.0  # High reward for reaching goal
            elif hit_obstacle:
                reward = -5.0  # Penalty for hitting obstacle
            elif timeout:
                reward = -10.0  # Penalty for timeout
            else:
                # Encourage movement toward the goal with a distance-based component
                distance_change = prev_distance - current_distance
                
                # Base step penalty to encourage efficiency
                step_penalty = -0.1
                
                # Bonus or penalty based on whether we moved closer to or further from the goal
                distance_reward = distance_change * 1.0
                
                reward = step_penalty + distance_reward
        
        # Get observation
        observation = self._get_observation()
        
        # Get info
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, goal_reached, timeout, info
    
    def _init_rendering(self) -> None:
        """Initialize rendering objects."""
        try:
            import pygame
            pygame.init()
            
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Navigation Environment")
            self.clock = pygame.time.Clock()
        except ImportError:
            self.render_mode = None
            print("pygame not installed, rendering disabled. Run `pip install pygame` to enable.")
    
    def _render_frame(self) -> Optional[np.ndarray]:
        """Render the current state of the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode is None:
            return None
        
        # For RGB array rendering without pygame
        if self.render_mode == "rgb_array":
            return self._render_array()
        
        # For human rendering with pygame
        elif self.render_mode == "human":
            try:
                import pygame
                
                if self.window is None:
                    self._init_rendering()
                
                cell_size = self.window_size // self.grid_size
                
                # Clear the screen
                self.window.fill((255, 255, 255))
                
                # Draw grid
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        rect = pygame.Rect(
                            j * cell_size,
                            i * cell_size,
                            cell_size,
                            cell_size
                        )
                        
                        # Draw cell color (obstacle or free)
                        if self.grid[i, j] == 1:
                            pygame.draw.rect(self.window, (0, 0, 0), rect)  # Black for obstacles
                        else:
                            pygame.draw.rect(self.window, (200, 200, 200), rect)  # Gray for free space
                        
                        # Draw cell border
                        pygame.draw.rect(self.window, (150, 150, 150), rect, 1)
                
                # Draw goal
                goal_rect = pygame.Rect(
                    self.goal_pos[1] * cell_size,
                    self.goal_pos[0] * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.window, (0, 255, 0), goal_rect)  # Green for goal
                
                # Draw agent
                agent_rect = pygame.Rect(
                    self.agent_pos[1] * cell_size,
                    self.agent_pos[0] * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.window, (255, 0, 0), agent_rect)  # Red for agent
                
                pygame.display.flip()
                
                # Add a delay
                self.clock.tick(self.metadata["render_fps"])
                return None
                
            except ImportError:
                self.render_mode = None
                print("pygame not installed, rendering disabled. Run `pip install pygame` to enable.")
                return None
    
    def _render_array(self) -> np.ndarray:
        """Render the environment as a RGB array using matplotlib.
        
        Returns:
            RGB array representing the environment.
        """
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # Create a copy of the grid for visualization
        vis_grid = self.grid.copy()
        
        # Mark the goal position
        vis_grid = np.where(vis_grid == 0, 0.3, 0.0)  # Free space as gray
        
        # Mark obstacles as black
        vis_grid = np.where(self.grid == 1, 0.0, vis_grid)
        
        # Mark goal position as green
        i, j = self.goal_pos
        vis_grid[i, j] = 0.8  # Goal as light green
        
        # Mark agent position as red
        i, j = self.agent_pos
        vis_grid[i, j] = 0.5  # Agent as red
        
        # Display grid
        ax.imshow(vis_grid, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1))
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        
        # Get RGB array from the plot
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        return self._render_frame()
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.window is not None:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass
            self.window = None
            self.clock = None 