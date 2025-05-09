# Environment Navigation with Depth-Estimated Terrain

This repo is horrible. Most of the scripts are me troubleshooting. Don't look at them. I gave up cleaning them. 

To run the code for yourself, go to the script.md file, read the comments, and run the commands line by line. 

Everything should work. hopefully. Let me know if it doesn't. Or GPT (probably GPT is gonna be more helpful).

A* should be instant. Q-learning will take about 5 min minimum for 64x64 grid. 

Note: It will take some time and ~1GB to download the model that converts image to depth map.



## ------Some  more markdown shenanigans------

This project implements a reinforcement learning system that:
1. Generates depth maps from single RGB images
2. Converts depth maps to navigable grid environments
3. Applies path planning algorithms to navigate through these reconstructed environments
4. Visualizes the entire process

## Project Structure

```
.
├── data/                     # Input data and intermediate results
│   ├── images/               # Input RGB images
│   ├── depth_maps/           # Generated depth maps
│   └── grid_maps/            # Processed grid environments
├── models/                   # Saved models
│   ├── depth/                # Pretrained depth estimation models
│   └── rl/                   # Trained RL agent models
├── src/                      # Source code
│   ├── depth_estimation/     # Depth map generation module
│   ├── grid_conversion/      # Depth to grid conversion module
│   ├── environments/         # RL environment implementation
│   ├── agents/               # RL agents implementation
│   ├── planning/             # Path planning algorithms
│   ├── visualization/        # Visualization utilities
│   └── utils/                # Common utilities
├── tests/                    # Test suite
├── notebooks/                # Jupyter notebooks for exploration
├── configs/                  # Configuration files
├── results/                  # Experimental results
├── scripts/                  # Utility scripts
├── pyproject.toml            # Project dependencies
├── setup.py                  # Package setup
├── .gitignore                # Git ignore patterns
├── LICENSE                   # Project license
└── README.md                 # This file
```

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

## Required Dependencies

- PyTorch: Deep learning framework
- OpenCV: Image processing
- Gymnasium: Reinforcement learning environments
- TIMM: PyTorch image models
- NumPy: Numerical computing
- Matplotlib: Visualization
- Pygame: Optional for interactive rendering

## Usage

### Depth Estimation

```python
from src.depth_estimation.midas import estimate_depth

# Generate depth map from image
image_path = "data/images/sample.jpg"
original_img, depth_map = estimate_depth(image_path)
```

### Grid Conversion

```python
from src.grid_conversion.converter import depth_to_grid, add_start_goal_points

# Convert depth map to navigable grid
grid = depth_to_grid(depth_map, grid_size=32, threshold_factor=0.6)

# Add start and goal points
grid_with_points = add_start_goal_points(grid, start_pos=(0, 0), goal_pos=(31, 31))
```

### Path Planning

```python
from src.planning.a_star import a_star_search, bidirectional_a_star
from src.visualization.env_visualizer import visualize_path

# Find optimal path using A* search
path, metrics = a_star_search(
    grid=grid,
    start=(0, 0),
    goal=(31, 31),
    heuristic="manhattan",  # Options: "manhattan", "euclidean", "chebyshev"
    diagonal=False          # Whether to allow diagonal movement
)

print(f"Path found with {len(path)} steps")
print(f"Nodes explored: {metrics['nodes_explored']}")
print(f"Time taken: {metrics['time_taken']:.4f} seconds")

# For larger grids, bidirectional A* can be more efficient
bidir_path, bidir_metrics = bidirectional_a_star(
    grid=grid,
    start=(0, 0),
    goal=(31, 31)
)

# Visualize the path in the environment
from src.environments.navigation_env import NavigationEnv
env = NavigationEnv(grid=grid, start_pos=(0, 0), goal_pos=(31, 31))
visualize_path(env, path, title="A* Optimal Path")
```

### Navigation Environment

```python
from src.environments.navigation_env import NavigationEnv
from src.visualization.env_visualizer import visualize_trajectory

# Create a custom navigation environment
env = NavigationEnv(
    grid=grid,
    start_pos=(0, 0),
    goal_pos=(31, 31),
    max_steps=200,
    render_mode="human"  # "human", "rgb_array", or None
)

# Reset the environment
observation, info = env.reset()

# Take steps in the environment
trajectory = [env.agent_pos]
rewards = []

for _ in range(100):
    # Take a random action
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    
    # Track trajectory and rewards
    trajectory.append(env.agent_pos)
    rewards.append(reward)
    
    if done or truncated:
        break

# Visualize the trajectory
visualize_trajectory(env, trajectory, rewards)
```

### Navigation with Reinforcement Learning

```python
from src.environments.navigation_env import NavigationEnv
from src.agents.q_learning import QLearningAgent
from src.visualization.env_visualizer import visualize_trajectory

# Create environment
env = NavigationEnv(
    grid=grid, 
    start_pos=(0, 0), 
    goal_pos=(31, 31),
    max_steps=200
)

# Create Q-learning agent
agent = QLearningAgent(
    action_space=env.action_space,
    grid_size=32,
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

# Visualize training progress
agent.plot_training_progress(window_size=50)

# Evaluate the agent
eval_results = agent.evaluate(
    env=env,
    num_episodes=100,
    show_progress=True
)

print(f"Success rate: {eval_results['success_rate'] * 100:.1f}%")
print(f"Average steps: {eval_results['avg_steps']:.1f}")

# Get and visualize the optimal path
optimal_path = agent.get_optimal_path(env)
visualize_trajectory(env, optimal_path, title="Q-learning Agent's Path")

# Save the trained agent
agent.save("models/rl/q_learning_agent.npz")

# Load a previously trained agent
loaded_agent = QLearningAgent.load("models/rl/q_learning_agent.npz", env.action_space)
```

### Visualization

```python
from src.visualization.visualizer import visualize_results
from src.visualization.grid_visualizer import visualize_depth_to_grid_comparison
from src.visualization.env_visualizer import create_trajectory_animation

# Visualize results
visualize_results(original_img, depth_map, grid, path)

# Visualize the transformation from RGB to depth to grid
visualize_depth_to_grid_comparison(original_img, depth_map, grid_with_points)

# Create an animation of an agent's trajectory
create_trajectory_animation(env, trajectory, save_path="results/trajectory.gif")
```

## Command Line Tools

### Depth Estimation

```bash
# Process a single image
python scripts/estimate_depth.py --image data/images/sample.jpg

# Process multiple images
python scripts/batch_depth_estimation.py --input-dir data/images --save-raw --save-colored --save-vis
```

### Grid Conversion

```bash
# Convert a depth map to a grid environment
python scripts/convert_depth_to_grid.py --input results/sample_depth.npy --grid-size 32 --threshold 0.6

# Batch process multiple depth maps
python scripts/batch_grid_conversion.py --input-dir results/depth_maps --save-visualization
```

### Path Planning

```bash
# Find and visualize the optimal path through a grid
python scripts/test_path_planning.py --grid results/grid_maps/sample_grid.npy --start-pos "0,0" --goal-pos "31,31"

# Generate a test grid and find optimal path
python scripts/test_path_planning.py --grid-size 32 --heuristic manhattan --diagonal

# Compare different path planning algorithms
python scripts/test_path_planning.py --grid-size 50 --compare
```

### Navigation Environment

```bash
# Test the navigation environment with a random agent
python scripts/test_navigation_env.py --grid-size 32 --steps 100 --save-animation

# Test with a specific grid or depth map
python scripts/test_navigation_env.py --grid results/grid_maps/sample_grid.npy
python scripts/test_navigation_env.py --depth results/depth_maps/sample_depth.npy
```

### Reinforcement Learning

```bash
# Train a Q-learning agent on a grid environment
python scripts/train_rl_agent.py --grid-size 10 --episodes 1000 --learning-rate 0.1 --discount-factor 0.95

# Train on a specific grid file
python scripts/train_rl_agent.py --grid results/grid_maps/sample_grid.npy --episodes 2000

# Train on a depth map
python scripts/train_rl_agent.py --depth results/depth_maps/sample_depth.npy --episodes 1500

# Save and evaluate the agent
python scripts/train_rl_agent.py --grid-size 15 --episodes 3000 --save-path "models/rl/custom_agent.npz" --eval-episodes 200
```

## Examples

### Complete Pipeline

```python
# 1. Generate depth map
rgb_image, depth_map = estimate_depth("data/images/sample.jpg")

# 2. Convert to grid
grid = depth_to_grid(depth_map, grid_size=32, threshold_factor=0.6)
grid_with_points = add_start_goal_points(grid)

# 3. Find optimal path through the grid
from src.planning.a_star import a_star_search
path, metrics = a_star_search(grid=grid, start=(0, 0), goal=(31, 31))

# 4. Create navigation environment and follow the path
env = NavigationEnv(grid=grid)
observation, _ = env.reset()
trajectory = [env.agent_pos]

for next_pos in path[1:]:  # Skip the first position (starting point)
    # Determine action to reach next position
    curr_row, curr_col = env.agent_pos
    next_row, next_col = next_pos
    
    if next_row < curr_row:
        action = 0  # Up
    elif next_col > curr_col:
        action = 1  # Right
    elif next_row > curr_row:
        action = 2  # Down
    else:
        action = 3  # Left
    
    # Take the action
    observation, reward, done, truncated, _ = env.step(action)
    trajectory.append(env.agent_pos)
    
    if done or truncated:
        break

# 5. Visualize the results
from src.visualization.env_visualizer import visualize_trajectory
visualize_trajectory(env, trajectory, title="A* Path Following")
```
