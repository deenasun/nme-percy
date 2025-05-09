"""Q-learning agent for grid navigation.

This module implements a tabular Q-learning agent for navigating through
grid environments created from depth maps.
"""

from typing import Dict, Tuple, List, Optional, Any, Union
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import deque

from src.environments.navigation_env import NavigationEnv
from src.visualization.rl_visualizer import (
    visualize_q_values as viz_q_values,
    visualize_max_q_values,
    visualize_policy,
    plot_training_progress as viz_training_progress,
    plot_exploration_rate as viz_exploration_rate,
    compare_paths
)


class QLearningAgent:
    """Q-learning agent for grid navigation.
    
    This class implements a tabular Q-learning agent that learns to navigate
    through grid environments by trial and error, gradually improving its
    performance through experience.
    
    Attributes:
        action_space: The action space of the environment
        grid_size: Size of the grid environment (assumed to be square)
        learning_rate: Step size parameter for Q-learning updates
        discount_factor: Discount factor for future rewards
        exploration_rate: Initial probability of random action (epsilon)
        exploration_decay: Rate at which exploration probability decays
        exploration_min: Minimum exploration rate
        q_table: Table of state-action values
    """
    
    def __init__(
        self, 
        action_space, 
        grid_size: int, 
        learning_rate: float = 0.1, 
        discount_factor: float = 0.95, 
        exploration_rate: float = 1.0, 
        exploration_decay: float = 0.995,
        exploration_min: float = 0.01
    ):
        """Initialize the Q-learning agent.
        
        Args:
            action_space: The action space of the environment
            grid_size: Size of the grid environment (assumed to be square)
            learning_rate: Step size parameter for Q-learning updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial probability of random action (epsilon)
            exploration_decay: Rate at which exploration probability decays
            exploration_min: Minimum exploration rate
        """
        self.action_space = action_space
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        
        # Initialize Q-table with zeros
        # States: position + goal
        self.q_table = np.zeros((grid_size, grid_size, grid_size, grid_size, action_space.n))
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
    
    def get_state_representation(self, observation: Dict) -> Tuple[int, int, int, int]:
        """Extract state representation from observation.
        
        Args:
            observation: Observation from the environment
            
        Returns:
            Tuple of (agent_row, agent_col, goal_row, goal_col)
        """
        # Handle both old and new observation dictionary formats
        if 'agent_position' in observation:
            pos_row, pos_col = observation['agent_position']
            goal_row, goal_col = observation['goal_position']
        elif 'position' in observation:
            pos_row, pos_col = observation['position']
            goal_row, goal_col = observation['goal']
        else:
            raise ValueError("Unknown observation format")
        
        return pos_row, pos_col, goal_row, goal_col
    
    def select_action(self, observation: Dict) -> int:
        """Select an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            Selected action ID
        """
        # Exploration-exploitation trade-off
        if random.random() < self.exploration_rate:
            return self.action_space.sample()  # Explore
        else:
            # State components
            pos_row, pos_col, goal_row, goal_col = self.get_state_representation(observation)
            
            # Ensure indices are within bounds of the Q-table
            pos_row = min(pos_row, self.grid_size - 1)
            pos_col = min(pos_col, self.grid_size - 1)
            goal_row = min(goal_row, self.grid_size - 1)
            goal_col = min(goal_col, self.grid_size - 1)
            
            # Exploit: select action with highest Q-value
            return np.argmax(self.q_table[pos_row, pos_col, goal_row, goal_col])
    
    def update(
        self, 
        observation: Dict, 
        action: int, 
        reward: float, 
        next_observation: Dict, 
        done: bool
    ) -> None:
        """Update Q-table based on the observed transition.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
        """
        # State components
        pos_row, pos_col, goal_row, goal_col = self.get_state_representation(observation)
        next_pos_row, next_pos_col, next_goal_row, next_goal_col = self.get_state_representation(next_observation)
        
        # Ensure indices are within bounds of the Q-table
        pos_row = min(pos_row, self.grid_size - 1)
        pos_col = min(pos_col, self.grid_size - 1)
        goal_row = min(goal_row, self.grid_size - 1)
        goal_col = min(goal_col, self.grid_size - 1)
        
        next_pos_row = min(next_pos_row, self.grid_size - 1)
        next_pos_col = min(next_pos_col, self.grid_size - 1)
        next_goal_row = min(next_goal_row, self.grid_size - 1)
        next_goal_col = min(next_goal_col, self.grid_size - 1)
        
        # Current Q-value
        current_q = self.q_table[pos_row, pos_col, goal_row, goal_col, action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_pos_row, next_pos_col, next_goal_row, next_goal_col])
        
        # Calculate new Q-value
        if done:
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[pos_row, pos_col, goal_row, goal_col, action] = new_q
    
    def decay_exploration(self) -> None:
        """Decay the exploration rate."""
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
    
    def train(
        self, 
        env: NavigationEnv, 
        num_episodes: int = 1000, 
        max_steps: int = None, 
        verbose: bool = True, 
        render_freq: int = 0,
        show_progress: bool = True,
        convergence_threshold: float = 0.01,
        convergence_episodes: int = 100,
        debug: bool = False
    ) -> Dict[str, List]:
        """Train the agent on the given environment.
        
        Args:
            env: The environment to train on
            num_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode (if None, uses env.max_steps)
            verbose: Whether to print training progress
            render_freq: How often to render the environment (0 = never)
            show_progress: Whether to show a progress bar
            convergence_threshold: Threshold for early stopping
            convergence_episodes: Number of episodes to check for convergence
            debug: Whether to print detailed debugging information
            
        Returns:
            Dictionary with training metrics
        """
        if max_steps is None:
            max_steps = env.max_steps
            
        # Create debug log file if debugging is enabled
        if debug:
            import os
            debug_dir = "debug_logs"
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = open(os.path.join(debug_dir, "q_learning_debug.log"), "w")
            debug_file.write("Episode,Step,State,Action,Reward,NextState,Done\n")
            
            # Log initial Q-table state
            debug_file.write(f"Initial Q-table stats: min={self.q_table.min()}, max={self.q_table.max()}, mean={self.q_table.mean()}\n")
        
        # Initialize metrics tracking
        rewards_history = []
        steps_history = []
        success_history = []
        exploration_rates = [self.exploration_rate]
        
        # Reset episode tracking attributes
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        
        # Convergence tracking
        recent_rewards = deque(maxlen=convergence_episodes)
        
        # Progress bar
        if show_progress:
            pbar = tqdm(total=num_episodes)
        
        # Main training loop
        for episode in range(num_episodes):
            # Reset environment
            observation, info = env.reset()
            
            if debug:
                debug_file.write(f"\nEpisode {episode+1}\n")
                debug_file.write(f"Start position: {env.agent_pos}, Goal position: {env.goal_pos}\n")
                debug_file.write(f"Exploration rate: {self.exploration_rate:.4f}\n")
            
            # Initialize episode variables
            episode_reward = 0
            episode_steps = 0
            done = False
            truncated = False
            
            # Episode loop
            while not (done or truncated) and episode_steps < max_steps:
                # Select action
                action = self.select_action(observation)
                
                # Take step in environment
                next_observation, reward, done, truncated, info = env.step(action)
                
                # Extract state representations for debugging
                if debug:
                    current_state = self.get_state_representation(observation)
                    next_state = self.get_state_representation(next_observation)
                    debug_file.write(f"Step {episode_steps}: State={current_state}, Action={action}, ")
                    debug_file.write(f"Reward={reward}, NextState={next_state}, Done={done}\n")
                
                # Update Q-table
                self.update(observation, action, reward, next_observation, done)
                
                # Render if requested
                if render_freq > 0 and episode % render_freq == 0:
                    env.render()
                
                # Update for next iteration
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
            
            # End of episode
            episode_success = done and not truncated  # Success if done (reached goal) and not truncated (out of steps)
            
            if debug:
                debug_file.write(f"Episode {episode+1} ended: Steps={episode_steps}, ")
                debug_file.write(f"Reward={episode_reward}, Success={episode_success}\n")
                
                # Log Q-table stats every 100 episodes
                if (episode + 1) % 100 == 0:
                    debug_file.write(f"Q-table stats after {episode+1} episodes: ")
                    debug_file.write(f"min={self.q_table.min()}, max={self.q_table.max()}, ")
                    debug_file.write(f"mean={self.q_table.mean()}, non-zero={(self.q_table != 0).sum()}\n")
            
            # Update metrics
            rewards_history.append(episode_reward)
            steps_history.append(episode_steps)
            success_history.append(float(episode_success))
            recent_rewards.append(episode_reward)
            
            # Store metrics in agent attributes for later use
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.success_rate.append(float(episode_success))
            
            # Decay exploration rate
            self.decay_exploration()
            exploration_rates.append(self.exploration_rate)
            
            # Update progress bar
            if show_progress:
                pbar.update(1)
                pbar.set_description(f"Episode {episode+1}/{num_episodes}: "
                                     f"Reward={episode_reward:.2f}, "
                                     f"Steps={episode_steps}, "
                                     f"Success={int(episode_success)}, "
                                     f"Explore={self.exploration_rate:.2f}")
            
            # Print if verbose
            elif verbose and (episode + 1) % 100 == 0:
                success_rate = np.mean(success_history[-100:]) if success_history else 0
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Steps={episode_steps}, "
                      f"Success Rate={success_rate:.2f}, "
                      f"Exploration={self.exploration_rate:.2f}")
            
            # Check for convergence
            if len(recent_rewards) == convergence_episodes:
                # If rewards have converged and success rate is good, stop early
                if np.std(recent_rewards) < convergence_threshold and np.mean(success_history[-convergence_episodes:]) > 0.95:
                    if verbose:
                        print(f"Converged after {episode+1} episodes.")
                    break
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        # Close debug file if debugging was enabled
        if debug:
            # Final Q-table stats
            debug_file.write(f"\nFinal Q-table stats: ")
            debug_file.write(f"min={self.q_table.min()}, max={self.q_table.max()}, ")
            debug_file.write(f"mean={self.q_table.mean()}, non-zero={(self.q_table != 0).sum()}\n")
            debug_file.close()
        
        # Return metrics
        return {
            'rewards': rewards_history,
            'steps': steps_history,
            'success_rate': success_history,
            'exploration_rates': exploration_rates
        }
    
    def evaluate(
        self, 
        env: NavigationEnv, 
        num_episodes: int = 100, 
        max_steps: int = None, 
        render_freq: int = 0,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Evaluate the agent on the environment.
        
        Args:
            env: The environment to evaluate on
            num_episodes: Number of episodes to evaluate for
            max_steps: Maximum number of steps per episode (if None, uses env's max_steps)
            render_freq: How often to render the environment (0 = never)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if max_steps is None:
            max_steps = env.max_steps
        
        # Performance tracking
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # Store original exploration rate and set to 0 for evaluation
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0  # No exploration during evaluation
        
        # Create iterator with progress bar if requested
        episodes = tqdm(range(num_episodes)) if show_progress else range(num_episodes)
        
        for episode in episodes:
            # Reset the environment
            observation, info = env.reset()
            total_reward = 0
            steps = 0
            done = False
            truncated = False
            
            # Render the environment if requested
            if render_freq > 0 and episode % render_freq == 0:
                if hasattr(env, 'render_mode') and env.render_mode is None:
                    print("Warning: Render mode is None, cannot render")
                else:
                    env.render()
            
            # Episode loop
            while not (done or truncated) and steps < max_steps:
                # Select action (no exploration)
                action = self.select_action(observation)
                
                # Take action in the environment
                next_observation, reward, done, truncated, info = env.step(action)
                
                # Update state and tracking variables
                observation = next_observation
                total_reward += reward
                steps += 1
                
                # Render the environment if requested
                if render_freq > 0 and episode % render_freq == 0:
                    if hasattr(env, 'render_mode') and env.render_mode is None:
                        pass
                    else:
                        env.render()
            
            # Update performance tracking
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            if done and not truncated:  # Goal reached
                success_count += 1
            
            # Update progress bar
            if show_progress:
                episodes.set_description(f"Evaluation {episode+1}/{num_episodes}")
                episodes.set_postfix({
                    'Reward': f"{total_reward:.2f}", 
                    'Steps': steps, 
                    'Success': done and not truncated
                })
        
        # Restore original exploration rate
        self.exploration_rate = original_exploration_rate
        
        # Calculate metrics
        avg_reward = sum(episode_rewards) / num_episodes
        avg_steps = sum(episode_lengths) / num_episodes
        success_rate = success_count / num_episodes
        
        # Return evaluation metrics
        return {
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_count': success_count
        }
    
    def get_optimal_path(
        self, 
        env: NavigationEnv, 
        max_steps: int = None
    ) -> List[Tuple[int, int]]:
        """Get the optimal path according to the learned Q-table.
        
        Args:
            env: The environment to navigate through
            max_steps: Maximum number of steps (if None, uses env's max_steps)
            
        Returns:
            List of (row, col) positions representing the optimal path
        """
        if max_steps is None:
            max_steps = env.max_steps
        
        # Store original exploration rate and set to 0 for optimal path
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0  # No exploration
        
        # Reset the environment to ensure we start at the correct position
        observation, info = env.reset()
        
        # Get the starting position from the observation
        if 'agent_position' in observation:
            start_pos = tuple(observation['agent_position'])
        elif 'position' in observation:
            start_pos = tuple(observation['position'])
        else:
            # Fallback to environment's agent position
            start_pos = env.agent_pos if hasattr(env, 'agent_pos') else (0, 0)
        
        # Cache the start position for later comparison
        env_start_pos = env.agent_pos if hasattr(env, 'agent_pos') else start_pos
        
        # Initialize path with starting position
        path = [start_pos]
        
        # Execute steps until done or max steps reached
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < max_steps:
            # Select action (no exploration)
            action = self.select_action(observation)
            
            # Take action in the environment
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Add position to path
            if 'agent_position' in next_observation:
                next_pos = tuple(next_observation['agent_position'])
            elif 'position' in next_observation:
                next_pos = tuple(next_observation['position'])
            else:
                # Fallback approach
                print("Warning: Could not extract position from observation")
                # Try to infer the next position based on the action
                row, col = path[-1]
                if action == 0:  # Up
                    next_pos = (max(0, row - 1), col)
                elif action == 1:  # Right
                    next_pos = (row, min(env.grid_size - 1, col + 1))
                elif action == 2:  # Down
                    next_pos = (min(env.grid_size - 1, row + 1), col)
                else:  # Left
                    next_pos = (row, max(0, col - 1))
            
            path.append(next_pos)
            
            # Update state and step counter
            observation = next_observation
            steps += 1
            
            # Break if we reached the goal
            if hasattr(env, 'goal_pos') and next_pos == env.goal_pos:
                done = True
        
        # Restore original exploration rate
        self.exploration_rate = original_exploration_rate
        
        # Check if the path doesn't start at the start position (this would indicate an issue)
        if path[0] != env_start_pos:
            print(f"Warning: Path does not start at environment's agent position. "
                  f"Path starts at {path[0]}, env.agent_pos is {env.agent_pos}")
        
        return path
    
    def plot_training_progress(
        self, 
        window_size: int = 100, 
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """Plot the training progress of the agent.
        
        Args:
            window_size: Window size for rolling average
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the plot (None = don't save)
            show: Whether to display the plot
        """
        # Check if training data exists
        if not self.episode_rewards or not self.episode_lengths:
            print("No training data available. Train the agent first.")
            return
        
        # Calculate success rate over time if not already available
        success_rate_smoothed = None
        if self.success_rate:
            # Convert success rate to moving average
            success_rate_smoothed = np.convolve(
                self.success_rate, 
                np.ones(window_size)/window_size, 
                mode='valid'
            ).tolist()
            
        # Use the visualization module to plot progress
        viz_training_progress(
            episode_rewards=self.episode_rewards,
            episode_steps=self.episode_lengths,
            success_rate=success_rate_smoothed,
            window_size=window_size,
            title="Q-Learning Training Progress",
            figsize=figsize,
            save_path=save_path,
            show=show
        )
    
    def plot_exploration_rate(
        self,
        title: str = "Exploration Rate Decay",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """Plot the exploration rate decay over time.
        
        Args:
            title: Title for the plot
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the plot (None = don't save)
            show: Whether to display the plot
        """
        # Check if we have episode rewards data
        if not self.episode_rewards:
            print("No training data available. Train the agent first.")
            return
            
        # Generate exploration rates
        num_episodes = len(self.episode_rewards)
        exploration_rates = [
            self.exploration_rate * (self.exploration_decay ** i) 
            for i in range(num_episodes)
        ]
        exploration_rates = [min(max(rate, self.exploration_min), 1.0) for rate in exploration_rates]
        
        # Use the visualization module to plot exploration rate
        viz_exploration_rate(
            exploration_rates=exploration_rates,
            title=title,
            figsize=figsize,
            save_path=save_path,
            show=show
        )
    
    def visualize_q_values(
        self, 
        env: NavigationEnv,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """Visualize the Q-values for the environment.
        
        Args:
            env: The environment to visualize Q-values for
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the visualization (None = don't save)
            show: Whether to display the visualization
        """
        # Extract Q-values for visualization
        goal_row, goal_col = env.goal_pos
        
        # Create a 3D array of Q-values for each position and action
        q_values = np.zeros((env.grid_size, env.grid_size, self.action_space.n))
        
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                q_values[row, col] = self.q_table[row, col, goal_row, goal_col]
        
        # Use the visualization module to plot Q-values
        viz_q_values(
            q_table=q_values,
            grid=env.grid,
            start_pos=env.agent_pos,
            goal_pos=env.goal_pos,
            title="Q-Values for Each Action",
            fig_size=figsize,
            save_path=save_path,
            show=show
        )
    
    def visualize_max_q_values(
        self,
        env: NavigationEnv,
        title: str = "Maximum Q-Values",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """Visualize the maximum Q-value for each state.
        
        Args:
            env: The environment to visualize Q-values for
            title: Title for the visualization
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the visualization (None = don't save)
            show: Whether to display the visualization
        """
        # Extract Q-values for visualization
        goal_row, goal_col = env.goal_pos
        
        # Create a 3D array of Q-values for each position and action
        q_values = np.zeros((env.grid_size, env.grid_size, self.action_space.n))
        
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                q_values[row, col] = self.q_table[row, col, goal_row, goal_col]
        
        # Use the visualization module to plot max Q-values
        visualize_max_q_values(
            q_table=q_values,
            grid=env.grid,
            start_pos=env.agent_pos,
            goal_pos=env.goal_pos,
            title=title,
            fig_size=figsize,
            save_path=save_path,
            show=show
        )
    
    def visualize_policy(
        self,
        env: NavigationEnv,
        title: str = "Learned Policy",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """Visualize the learned policy (best action) for each state.
        
        Args:
            env: The environment to visualize the policy for
            title: Title for the visualization
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the visualization (None = don't save)
            show: Whether to display the visualization
        """
        # Extract Q-values for visualization
        goal_row, goal_col = env.goal_pos
        
        # Create a 3D array of Q-values for each position and action
        q_values = np.zeros((env.grid_size, env.grid_size, self.action_space.n))
        
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                q_values[row, col] = self.q_table[row, col, goal_row, goal_col]
        
        # Use the visualization module to plot policy
        visualize_policy(
            q_table=q_values,
            grid=env.grid,
            start_pos=env.agent_pos,
            goal_pos=env.goal_pos,
            title=title,
            fig_size=figsize,
            save_path=save_path,
            show=show
        )
    
    def compare_with_path(
        self,
        env: NavigationEnv,
        reference_path: List[Tuple[int, int]],
        q_learning_label: str = "Q-Learning Path",
        reference_label: str = "Reference Path",
        title: str = "Path Comparison",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> None:
        """Compare the Q-learning path with a reference path.
        
        Args:
            env: The navigation environment
            reference_path: The reference path to compare with
            q_learning_label: Label for the Q-learning path
            reference_label: Label for the reference path
            title: Title for the plot
            figsize: Size of the figure (width, height) in inches
            save_path: Path to save the visualization (None = don't save)
            show: Whether to display the visualization
        """
        # Get the Q-learning path
        q_learning_path = self.get_optimal_path(env)
        
        # Use the visualization module to compare paths
        compare_paths(
            env=env,
            q_learning_path=q_learning_path,
            reference_path=reference_path,
            q_learning_label=q_learning_label,
            reference_label=reference_label,
            title=title,
            fig_size=figsize,
            save_path=save_path,
            show=show
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the agent's Q-table and parameters to a file.
        
        Args:
            path: Path to save the agent
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save agent data
        np.savez(
            path,
            q_table=self.q_table,
            grid_size=self.grid_size,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            exploration_rate=self.exploration_rate,
            exploration_decay=self.exploration_decay,
            exploration_min=self.exploration_min,
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            success_rate=self.success_rate
        )
        print(f"Agent saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], action_space) -> 'QLearningAgent':
        """Load an agent from a file.
        
        Args:
            path: Path to load the agent from
            action_space: Action space of the environment
            
        Returns:
            Loaded Q-learning agent
        """
        # Load data
        data = np.load(path, allow_pickle=True)
        
        # Create a new agent
        agent = cls(
            action_space=action_space,
            grid_size=int(data['grid_size']),
            learning_rate=float(data['learning_rate']),
            discount_factor=float(data['discount_factor']),
            exploration_rate=float(data['exploration_rate']),
            exploration_decay=float(data['exploration_decay']),
            exploration_min=float(data['exploration_min'])
        )
        
        # Load Q-table and performance metrics
        agent.q_table = data['q_table']
        
        # Restore performance metrics if available
        if 'episode_rewards' in data:
            agent.episode_rewards = data['episode_rewards'].tolist()
        if 'episode_lengths' in data:
            agent.episode_lengths = data['episode_lengths'].tolist()
        if 'success_rate' in data:
            agent.success_rate = data['success_rate'].tolist()
        
        print(f"Agent loaded from {path}")
        return agent 