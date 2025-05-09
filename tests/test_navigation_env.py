"""Unit tests for the navigation environment."""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.environments.navigation_env import NavigationEnv


class TestNavigationEnv(unittest.TestCase):
    """Test cases for navigation environment functionality."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a simple test grid
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Add some obstacles
        self.grid[0, :] = 1  # Top wall
        self.grid[-1, :] = 1  # Bottom wall
        self.grid[:, 0] = 1  # Left wall
        self.grid[:, -1] = 1  # Right wall
        self.grid[5, 2:8] = 1  # Horizontal obstacle
        
        # Define start and goal positions
        self.start_pos = (1, 1)
        self.goal_pos = (8, 8)
        
        # Create environment
        self.env = NavigationEnv(
            grid=self.grid,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            max_steps=100,
            render_mode=None
        )
    
    def test_initialization(self) -> None:
        """Test environment initialization."""
        # Check grid
        np.testing.assert_array_equal(self.env.grid, self.grid)
        
        # Check positions
        self.assertEqual(self.env.start_pos, self.start_pos)
        self.assertEqual(self.env.goal_pos, self.goal_pos)
        self.assertEqual(self.env.agent_pos, self.start_pos)
        
        # Check space dimensions
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(
            self.env.observation_space['grid'].shape, 
            (self.grid_size, self.grid_size)
        )
        self.assertEqual(self.env.observation_space['position'].shape, (2,))
        self.assertEqual(self.env.observation_space['goal'].shape, (2,))
    
    def test_reset(self) -> None:
        """Test environment reset."""
        # Take some actions to move the agent
        self.env.step(1)  # Right
        self.env.step(2)  # Down
        
        # Reset and check
        observation, info = self.env.reset()
        
        # Check agent position
        self.assertEqual(self.env.agent_pos, self.start_pos)
        np.testing.assert_array_equal(
            observation['position'], 
            np.array(self.start_pos, dtype=np.int32)
        )
        
        # Check info contains distance
        self.assertIn('distance_to_goal', info)
        
        # Check with custom seed
        observation, _ = self.env.reset(seed=42)
        self.assertEqual(self.env.agent_pos, self.start_pos)
        
        # Check with custom start position
        custom_start = (2, 2)
        observation, _ = self.env.reset(options={'start_pos': custom_start})
        self.assertEqual(self.env.agent_pos, custom_start)
    
    def test_step(self) -> None:
        """Test environment step function."""
        # Reset to ensure consistent state
        self.env.reset()
        
        # Test each action
        # Action 0: Up
        obs, reward, done, truncated, info = self.env.step(0)
        expected_pos = (max(0, self.start_pos[0] - 1), self.start_pos[1])
        if self.grid[expected_pos] == 1:  # If obstacle, position shouldn't change
            expected_pos = self.start_pos
        self.assertEqual(self.env.agent_pos, expected_pos)
        
        # Reset for next test
        self.env.reset()
        
        # Action 1: Right
        obs, reward, done, truncated, info = self.env.step(1)
        expected_pos = (self.start_pos[0], min(self.grid_size - 1, self.start_pos[1] + 1))
        if self.grid[expected_pos] == 1:  # If obstacle, position shouldn't change
            expected_pos = self.start_pos
        self.assertEqual(self.env.agent_pos, expected_pos)
        
        # Reset for next test
        self.env.reset()
        
        # Action 2: Down
        obs, reward, done, truncated, info = self.env.step(2)
        expected_pos = (min(self.grid_size - 1, self.start_pos[0] + 1), self.start_pos[1])
        if self.grid[expected_pos] == 1:  # If obstacle, position shouldn't change
            expected_pos = self.start_pos
        self.assertEqual(self.env.agent_pos, expected_pos)
        
        # Reset for next test
        self.env.reset()
        
        # Action 3: Left
        obs, reward, done, truncated, info = self.env.step(3)
        expected_pos = (self.start_pos[0], max(0, self.start_pos[1] - 1))
        if self.grid[expected_pos] == 1:  # If obstacle, position shouldn't change
            expected_pos = self.start_pos
        self.assertEqual(self.env.agent_pos, expected_pos)
        
        # Check observation
        np.testing.assert_array_equal(obs['grid'], self.grid)
        np.testing.assert_array_equal(obs['position'], np.array(self.env.agent_pos, dtype=np.int32))
        np.testing.assert_array_equal(obs['goal'], np.array(self.goal_pos, dtype=np.int32))
        
        # Check info
        self.assertIn('distance_to_goal', info)
        self.assertIn('steps', info)
    
    def test_goal_reached(self) -> None:
        """Test goal state termination and reward."""
        # Create an environment with agent starting next to the goal
        grid = np.zeros((5, 5), dtype=np.int8)
        start_pos = (2, 2)
        goal_pos = (2, 3)  # Adjacent to start
        
        env = NavigationEnv(
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            max_steps=10,
            render_mode=None
        )
        
        # Reset and take action to reach goal
        env.reset()
        _, reward, done, truncated, _ = env.step(1)  # Move right to goal
        
        # Check termination and reward
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(reward, 100)  # Large positive reward for reaching goal
    
    def test_obstacle_collision(self) -> None:
        """Test obstacle collision termination and reward."""
        # Create environment with agent starting next to an obstacle
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[2, 3] = 1  # Obstacle to the right of start
        start_pos = (2, 2)
        goal_pos = (4, 4)
        
        env = NavigationEnv(
            grid=grid,
            start_pos=start_pos,
            goal_pos=goal_pos,
            max_steps=10,
            render_mode=None
        )
        
        # Reset and take action to hit obstacle
        env.reset()
        
        # Store initial position
        initial_pos = env.agent_pos
        
        # Move right into obstacle
        _, reward, done, truncated, _ = env.step(1)
        
        # Check position (should stay the same)
        self.assertEqual(env.agent_pos, initial_pos)
        
        # Check we didn't terminate (just hitting an obstacle doesn't terminate)
        self.assertFalse(done)
        self.assertFalse(truncated)
        
        # Default reward should be -1 per step
        self.assertEqual(reward, -1)
    
    def test_max_steps(self) -> None:
        """Test truncation after max steps."""
        # Create environment with a small max_steps value
        max_steps = 5
        env = NavigationEnv(
            grid=self.grid,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            max_steps=max_steps,
            render_mode=None
        )
        
        # Reset and take exactly max_steps
        env.reset()
        
        for i in range(max_steps):
            _, _, done, truncated, _ = env.step(0)  # Take some action
            if i < max_steps - 1:
                self.assertFalse(truncated)
            else:
                # On the last step, should be truncated
                self.assertTrue(truncated)
    
    def test_custom_reward_fn(self) -> None:
        """Test custom reward function."""
        # Define a custom reward function
        def custom_reward_fn(grid, position, goal, is_done, step_count):
            # Distance-based reward
            distance = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
            return -distance  # Negative distance as reward
        
        # Create environment with custom reward function
        env = NavigationEnv(
            grid=self.grid,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            max_steps=10,
            reward_fn=custom_reward_fn,
            render_mode=None
        )
        
        # Reset and take a step
        env.reset()
        _, reward, _, _, _ = env.step(1)  # Move right
        
        # Calculate expected reward
        position = env.agent_pos
        expected_reward = -(abs(position[0] - self.goal_pos[0]) + abs(position[1] - self.goal_pos[1]))
        
        # Check reward
        self.assertEqual(reward, expected_reward)
    
    def test_invalid_grid(self) -> None:
        """Test error handling for invalid grid."""
        # 1D grid
        with self.assertRaises(ValueError):
            NavigationEnv(grid=np.zeros(10), start_pos=(0, 0), goal_pos=(9, 9))
        
        # 3D grid
        with self.assertRaises(ValueError):
            NavigationEnv(grid=np.zeros((5, 5, 3)), start_pos=(0, 0), goal_pos=(4, 4))
    
    def test_invalid_positions(self) -> None:
        """Test error handling for invalid positions."""
        grid = np.zeros((5, 5))
        
        # Out of bounds start position
        with self.assertRaises(ValueError):
            NavigationEnv(grid=grid, start_pos=(-1, 0), goal_pos=(4, 4))
        
        # Out of bounds goal position
        with self.assertRaises(ValueError):
            NavigationEnv(grid=grid, start_pos=(0, 0), goal_pos=(5, 5))
        
        # Start position on obstacle
        grid[0, 0] = 1
        with self.assertRaises(ValueError):
            NavigationEnv(grid=grid, start_pos=(0, 0), goal_pos=(4, 4))
        
        # Goal position on obstacle
        grid[0, 0] = 0
        grid[4, 4] = 1
        with self.assertRaises(ValueError):
            NavigationEnv(grid=grid, start_pos=(0, 0), goal_pos=(4, 4))


if __name__ == "__main__":
    unittest.main() 