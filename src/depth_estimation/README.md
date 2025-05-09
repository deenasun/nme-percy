# Depth Estimation Module

This module implements monocular depth estimation using the MiDaS model from Intel ISL. It provides functionality to generate depth maps from single RGB images.

## Overview

The depth estimation module consists of the following components:

- `midas.py`: Main implementation of the MiDaS depth estimation model
- `utils.py`: Utility functions for processing and saving depth maps

## Usage

### Basic Usage

```python
from src.depth_estimation.midas import estimate_depth

# Generate depth map from image
image_path = "data/images/sample.jpg"
original_img, depth_map = estimate_depth(image_path)
```

### Advanced Usage

```python
from src.depth_estimation.midas import DepthEstimator
from src.depth_estimation.utils import save_depth_visualization

# Create depth estimator with specific model type
estimator = DepthEstimator(model_type="DPT_Large")

# Estimate depth
rgb_image, depth_map = estimator.estimate_depth("data/images/sample.jpg")

# Visualize results
estimator.visualize_depth(rgb_image, depth_map, save_path="results/sample_depth.png")

# Save visualization with custom settings
save_depth_visualization(
    rgb_image, 
    depth_map, 
    "results/sample_custom_vis.png",
    colormap="plasma"
)
```

## MiDaS Model Variants

MiDaS offers three model variants:

1. **MiDaS_small**: Faster inference but lower accuracy
2. **DPT_Large**: Highest accuracy but slower inference
3. **DPT_Hybrid**: Balance between speed and accuracy

For a proof-of-concept or testing, it's recommended to use MiDaS_small. For final results or presentations, DPT_Large or DPT_Hybrid will provide better quality depth maps.

## Dependencies

The module requires the following dependencies:

- PyTorch (for the MiDaS model)
- OpenCV (for image processing)
- NumPy (for numerical operations)
- Matplotlib (for visualization)

All dependencies are listed in the project's `pyproject.toml` file.

## Internet Requirements

When you first run the module, it will download the pre-trained MiDaS model from PyTorch Hub. This requires an internet connection for the initial setup, but once downloaded, the model can be used offline. 