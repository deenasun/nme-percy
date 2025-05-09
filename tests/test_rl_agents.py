#!/usr/bin/env python
"""Unit tests for reinforcement learning agents.

This module contains tests for the reinforcement learning agents,
focusing on the Q-learning implementation.
"""

import unittest
import sys
import tempfile
import os
import numpy as np
from pathlib import Path
from gymnasium import spaces

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agents.q_learning import QLearningAgent
from src.environments.navigation_env import NavigationEnv


class TestQLearningAgent(unittest.TestCase):
    """Test cases for the Q-learning agent."""

    def setUp(self):
        """Set up test fixtures for each test method."""
        # Create a simple grid for testing
        self.grid_size = 5
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.grid[2, 2] = 1  # Add an obstacle
        
        # Create an environment
        self.env = NavigationEnv(
            grid=self.grid,
            start_pos=(0, 0),
            goal_pos=(4, 4),
            max_steps=20
        )
        
        # Create an agent
        self.agent = QLearningAgent(
            action_space=self.env.action_space,
            grid_size=self.grid_size,
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.5,
            exploration_decay=0.99,
            exploration_min=0.01
        )

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.grid_size, self.grid_size)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.95)
        self.assertEqual(self.agent.exploration_rate, 0.5)
        self.assertEqual(self.agent.exploration_decay, 0.99)
        self.assertEqual(self.agent.exploration_min, 0.01)
        
        # Check Q-table shape
        expected_shape = (self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.env.action_space.n)
        self.assertEqual(self.agent.q_table.shape, expected_shape)
        
        # Check if Q-table is initialized with zeros
        self.assertTrue(np.all(self.agent.q_table == 0))

    def test_get_state_representation(self):
        """Test extracting state representation from observation."""
        observation = {
            'position': (1, 2),
            'goal': (3, 4)
        }
        
        state_rep = self.agent.get_state_representation(observation)
        self.assertEqual(state_rep, (1, 2, 3, 4))

    def test_select_action_explore(self):
        """Test action selection with exploration."""
        # Force exploration
        self.agent.exploration_rate = 1.0
        
        observation = {
            'position': (0, 0),
            'goal': (4, 4)
        }
        
        # Actions should be random, within action space
        for _ in range(10):
            action = self.agent.select_action(observation)
            self.assertIn(action, range(4))  # 4 actions: up, right, down, left

    def test_select_action_exploit(self):
        """Test action selection with exploitation."""
        # Force exploitation
        self.agent.exploration_rate = 0.0
        
        observation = {
            'position': (0, 0),
            'goal': (4, 4)
        }
        
        # Set Q-values for this state
        self.agent.q_table[0, 0, 4, 4, 0] = 0.1  # Up
        self.agent.q_table[0, 0, 4, 4, 1] = 0.3  # Right
        self.agent.q_table[0, 0, 4, 4, 2] = 0.0  # Down
        self.agent.q_table[0, 0, 4, 4, 3] = 0.2  # Left
        
        # Should choose action with highest Q-value (Right)
        action = self.agent.select_action(observation)
        self.assertEqual(action, 1)

    def test_update(self):
        """Test Q-value update."""
        observation = {
            'position': (1, 1),
            'goal': (4, 4)
        }
        action = 1  # Right
        reward = 0.5
        next_observation = {
            'position': (1, 2),
            'goal': (4, 4)
        }
        done = False
        
        # Initial Q-value
        initial_q = self.agent.q_table[1, 1, 4, 4, action]
        self.assertEqual(initial_q, 0.0)
        
        # Update Q-value
        self.agent.update(observation, action, reward, next_observation, done)
        
        # Check if Q-value was updated
        updated_q = self.agent.q_table[1, 1, 4, 4, action]
        self.assertGreater(updated_q, initial_q)
        
        # Expected Q-value calculation
        max_next_q = np.max(self.agent.q_table[1, 2, 4, 4])
        expected_q = initial_q + self.agent.learning_rate * (reward + self.agent.discount_factor * max_next_q - initial_q)
        self.assertAlmostEqual(updated_q, expected_q)

    def test_decay_exploration(self):
        """Test exploration rate decay."""
        initial_rate = self.agent.exploration_rate
        self.agent.decay_exploration()
        
        decayed_rate = self.agent.exploration_rate
        self.assertLess(decayed_rate, initial_rate)
        
        # Check if decay matches expected formula
        expected_rate = max(self.agent.exploration_min, initial_rate * self.agent.exploration_decay)
        self.assertEqual(decayed_rate, expected_rate)
        
        # Test minimum bound
        self.agent.exploration_rate = self.agent.exploration_min / 2
        self.agent.decay_exploration()
        self.assertEqual(self.agent.exploration_rate, self.agent.exploration_min)

    def test_train(self):
        """Test agent training."""
        # Use fewer episodes for faster testing
        num_episodes = 10
        
        # Train the agent
        results = self.agent.train(
            env=self.env,
            num_episodes=num_episodes,
            verbose=False,
            show_progress=False
        )
        
        # Check if performance tracking lists were updated
        self.assertEqual(len(self.agent.episode_rewards), num_episodes)
        self.assertEqual(len(self.agent.episode_lengths), num_episodes)
        self.assertEqual(len(self.agent.success_rate), num_episodes)
        
        # Check that training results were returned
        self.assertIn('episode_rewards', results)
        self.assertIn('episode_lengths', results)
        self.assertIn('success_rate', results)
        self.assertIn('training_time', results)
        
        # Check that Q-table was updated (not all zeros anymore)
        self.assertFalse(np.all(self.agent.q_table == 0))

    def test_evaluate(self):
        """Test agent evaluation."""
        # First train the agent a bit
        self.agent.train(
            env=self.env,
            num_episodes=10,
            verbose=False,
            show_progress=False
        )
        
        # Evaluate the agent
        eval_results = self.agent.evaluate(
            env=self.env,
            num_episodes=5,
            show_progress=False
        )
        
        # Check if evaluation results were returned
        self.assertIn('avg_reward', eval_results)
        self.assertIn('avg_steps', eval_results)
        self.assertIn('success_rate', eval_results)
        self.assertIn('episode_rewards', eval_results)
        self.assertIn('episode_lengths', eval_results)
        self.assertIn('success_count', eval_results)
        
        # Check that values are in expected ranges
        self.assertGreaterEqual(eval_results['success_rate'], 0)
        self.assertLessEqual(eval_results['success_rate'], 1)
        self.assertGreaterEqual(eval_results['avg_steps'], 1)

    def test_get_optimal_path(self):
        """Test getting optimal path."""
        # First train the agent a bit
        self.agent.train(
            env=self.env,
            num_episodes=10,
            verbose=False,
            show_progress=False
        )
        
        # Get optimal path
        path = self.agent.get_optimal_path(self.env)
        
        # Check path properties
        self.assertIsInstance(path, list)
        self.assertGreaterEqual(len(path), 1)
        
        # Check that the path starts at the start position
        self.assertEqual(path[0], (0, 0))
        
        # Check path continuity
        for i in range(1, len(path)):
            prev_pos = path[i-1]
            curr_pos = path[i]
            
            # Calculate Manhattan distance between consecutive positions
            manhattan_dist = abs(prev_pos[0] - curr_pos[0]) + abs(prev_pos[1] - curr_pos[1])
            
            # Consecutive positions should be adjacent (Manhattan distance = 1)
            self.assertEqual(manhattan_dist, 1)

    def test_save_load(self):
        """Test saving and loading agent."""
        # Create some non-zero Q-values
        self.agent.q_table[0, 0, 4, 4, 1] = 0.5
        self.agent.episode_rewards = [1.0, 2.0, 3.0]
        self.agent.episode_lengths = [10, 9, 8]
        self.agent.success_rate = [0, 0, 1]
        
        # Create a temporary file for saving
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            save_path = tmp.name
        
        try:
            # Save the agent
            self.agent.save(save_path)
            
            # Create a new agent with different parameters
            new_agent = QLearningAgent(
                action_space=self.env.action_space,
                grid_size=self.grid_size,
                learning_rate=0.2,  # Different from original
                discount_factor=0.9,  # Different from original
                exploration_rate=0.3  # Different from original
            )
            
            # Load saved agent
            loaded_agent = QLearningAgent.load(save_path, self.env.action_space)
            
            # Check that parameters were restored correctly
            self.assertEqual(loaded_agent.grid_size, self.agent.grid_size)
            self.assertEqual(loaded_agent.learning_rate, self.agent.learning_rate)
            self.assertEqual(loaded_agent.discount_factor, self.agent.discount_factor)
            self.assertEqual(loaded_agent.exploration_rate, self.agent.exploration_rate)
            
            # Check that Q-table was restored correctly
            self.assertEqual(loaded_agent.q_table[0, 0, 4, 4, 1], 0.5)
            
            # Check that performance metrics were restored
            self.assertEqual(loaded_agent.episode_rewards, [1.0, 2.0, 3.0])
            self.assertEqual(loaded_agent.episode_lengths, [10, 9, 8])
            self.assertEqual(loaded_agent.success_rate, [0, 0, 1])
            
        finally:
            # Clean up temporary file
            if os.path.exists(save_path):
                os.unlink(save_path)


if __name__ == "__main__":
    unittest.main() 