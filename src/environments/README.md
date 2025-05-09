# Navigation Environments

This module implements custom Gymnasium environments for navigating grid-based environments derived from depth maps. These environments can be used for reinforcement learning experiments and path planning.

## NavigationEnv

The main class is `NavigationEnv`, which implements a grid-based navigation environment where an agent must navigate from a start position to a goal position while avoiding obstacles.

### Features

- **Grid-Based Navigation**: Navigate through a 2D grid with obstacles
- **Customizable Start/Goal**: Define custom start and goal positions
- **Flexible Rewards**: Use the default reward structure or provide a custom reward function
- **Step Limit**: Set a maximum number of steps per episode
- **Visualization**: Built-in rendering through pygame or matplotlib

### Example Usage

```python
from src.environments.navigation_env import NavigationEnv
from src.grid_conversion.converter import depth_to_grid

# Create a grid from a depth map
grid = depth_to_grid(depth_map, grid_size=32, threshold_factor=0.6)

# Create the environment
env = NavigationEnv(
    grid=grid,
    start_pos=(0, 0),
    goal_pos=(31, 31),
    max_steps=200,
    render_mode="human"
)

# Reset the environment
obs, info = env.reset()

# Take a step
action = 1  # Move right
obs, reward, done, truncated, info = env.step(action)
```

### Observation Space

The observation is a dictionary containing:
- `grid`: The grid environment (2D array of 0s and 1s)
- `position`: The agent's position as (row, col)
- `goal`: The goal position as (row, col)

### Action Space

The action space is discrete with 4 actions:
- `0`: Move up
- `1`: Move right
- `2`: Move down
- `3`: Move left

### Reward Structure

The default reward structure is:
- `-1` for each step (encourages reaching the goal quickly)
- `+100` for reaching the goal
- `-100` for hitting an obstacle

A custom reward function can be provided during initialization.

### Rendering

The environment supports two rendering modes:
- `human`: Renders the environment using pygame for real-time visualization
- `rgb_array`: Returns an RGB array that can be used for offline visualization

## Example Visualization

The environment can be visualized using the visualization utilities in `src/visualization/env_visualizer.py`:

```python
from src.visualization.env_visualizer import visualize_trajectory, create_trajectory_animation

# Run a random agent
trajectory = []
rewards = []
obs, _ = env.reset()
done = False
truncated = False

while not (done or truncated):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, _ = env.step(action)
    trajectory.append(env.agent_pos)
    rewards.append(reward)

# Visualize the trajectory
visualize_trajectory(env, trajectory, rewards)

# Create an animation
create_trajectory_animation(env, trajectory)
```

## Integration with Other Modules

The environment is designed to work seamlessly with other modules in the project:

```python
# Complete pipeline from RGB image to depth map to grid to environment
from src.depth_estimation.midas import estimate_depth
from src.grid_conversion.converter import depth_to_grid

# Generate depth map
rgb_image, depth_map = estimate_depth("data/images/sample.jpg")

# Convert to grid
grid = depth_to_grid(depth_map, grid_size=32, threshold_factor=0.6)

# Create environment
env = NavigationEnv(grid=grid)
``` 