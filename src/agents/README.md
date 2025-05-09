# Reinforcement Learning Agents

This module contains implementations of reinforcement learning (RL) agents for navigation in grid environments.

## Q-Learning Agent

The Q-learning agent is a tabular reinforcement learning agent that learns to navigate through grid environments using the Q-learning algorithm.

### Features

- **Tabular Q-learning**: Stores and updates a table of state-action values (Q-values)
- **Epsilon-greedy exploration**: Balances exploration and exploitation with a decaying exploration rate
- **Performance tracking**: Tracks episode rewards, lengths, and success rates
- **Learning visualization**: Provides functions to visualize training progress and learned Q-values
- **Save and load**: Supports saving and loading trained agents

### Usage Example

```python
from src.environments.navigation_env import NavigationEnv
from src.agents.q_learning import QLearningAgent
from src.visualization.env_visualizer import visualize_trajectory

# Create a grid environment
grid = np.zeros((10, 10), dtype=np.int8)
grid[3:7, 5] = 1  # Add a vertical wall

# Create environment
env = NavigationEnv(
    grid=grid,
    start_pos=(0, 0),
    goal_pos=(9, 9),
    max_steps=100
)

# Create agent
agent = QLearningAgent(
    action_space=env.action_space,
    grid_size=10,
    learning_rate=0.1,
    discount_factor=0.95,
    exploration_rate=1.0,
    exploration_decay=0.995,
    exploration_min=0.01
)

# Train agent
training_results = agent.train(
    env=env,
    num_episodes=1000,
    verbose=True,
    show_progress=True
)

# Plot training progress
agent.plot_training_progress(window_size=50)

# Visualize learned Q-values
agent.visualize_q_values(env)

# Get the optimal path found by the agent
optimal_path = agent.get_optimal_path(env)

# Visualize the optimal path
visualize_trajectory(env, optimal_path, title="Q-learning Agent's Optimal Path")

# Save the trained agent
agent.save("models/rl/q_learning_agent.npz")

# Evaluate the agent
evaluation_results = agent.evaluate(
    env=env,
    num_episodes=100,
    show_progress=True
)

print(f"Success rate: {evaluation_results['success_rate'] * 100:.1f}%")
print(f"Average steps: {evaluation_results['avg_steps']:.1f}")
print(f"Average reward: {evaluation_results['avg_reward']:.2f}")
```

### Parameters

- **grid_size**: Size of the grid environment (assumed to be square)
- **learning_rate**: Step size parameter for Q-learning updates (alpha)
- **discount_factor**: Discount factor for future rewards (gamma)
- **exploration_rate**: Initial probability of random action (epsilon)
- **exploration_decay**: Rate at which exploration probability decays
- **exploration_min**: Minimum exploration rate

### State Representation

The agent represents states as a tuple of `(agent_row, agent_col, goal_row, goal_col)`, which allows it to learn navigation policies for different start and goal positions.

### Limitations

- **Memory usage**: The tabular approach requires storing Q-values for every state-action pair, which can be memory-intensive for large grids.
- **Fixed grid size**: The agent is initialized for a specific grid size and cannot easily adapt to different sized environments.
- **Discrete state space**: Only works with discrete states and actions.

## Improvements and Extensions

Future improvements to the agent could include:

- **Function approximation**: Replace the Q-table with a neural network for better scalability
- **Experience replay**: Store and reuse past experiences for more efficient learning
- **Prioritized sweeping**: Focus updates on states with large value changes
- **Double Q-learning**: Reduce overestimation of Q-values

Other types of agents that could be implemented include:

- **SARSA**: On-policy TD control
- **DQN**: Deep Q-Network for larger state spaces
- **Policy Gradient**: Direct policy optimization
- **Actor-Critic**: Combine value function and policy approaches 