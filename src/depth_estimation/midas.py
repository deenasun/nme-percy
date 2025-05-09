"""MiDaS monocular depth estimation implementation.

This module provides functionality to estimate depth maps from single RGB images
using the pre-trained MiDaS model from Intel ISL.
"""

from typing import Tuple, Optional, Union, Literal
import os
import logging

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Available model types
ModelType = Literal["MiDaS_small", "DPT_Large", "DPT_Hybrid"]

class DepthEstimator:
    """MiDaS depth estimation model wrapper.
    
    This class provides an interface to the MiDaS model for monocular depth estimation.
    """
    
    def __init__(
        self, 
        model_type: ModelType = "MiDaS_small", 
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize the depth estimator with specified model type.
        
        Args:
            model_type: The MiDaS model variant to use. Options are:
                - "MiDaS_small": Faster but less accurate
                - "DPT_Large": Highest accuracy but slower
                - "DPT_Hybrid": Balance between speed and accuracy
            device: The device to run inference on. If None, will use CUDA if available.
        """
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load MiDaS model
        try:
            logger.info(f"Loading MiDaS model: {model_type}")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load appropriate transforms
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "MiDaS_small":
                self.transform = self.transforms.small_transform
            else:
                self.transform = self.transforms.dpt_transform
            
            logger.info("MiDaS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MiDaS model: {e}")
            raise RuntimeError(f"Failed to load MiDaS model: {e}") from e

    def estimate_depth(
        self, 
        img_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a depth map from an input RGB image.
        
        Args:
            img_path: Path to the input RGB image.
            
        Returns:
            A tuple containing:
                - The original RGB image (numpy array)
                - The estimated depth map (numpy array)
                
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            RuntimeError: If depth estimation fails.
        """
        # Ensure the image file exists
        if not os.path.isfile(img_path):
            logger.error(f"Image file not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        try:
            # Load image
            logger.info(f"Processing image: {img_path}")
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"Failed to load image from {img_path}")
            
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Transform image for model
            input_batch = self.transform(img).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
            logger.info("Depth estimation completed successfully")
            return img, depth_map
            
        except Exception as e:
            logger.error(f"Error during depth estimation: {e}")
            raise RuntimeError(f"Depth estimation failed: {e}") from e
    
    def visualize_depth(
        self, 
        rgb_image: np.ndarray, 
        depth_map: np.ndarray, 
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        cmap: str = "inferno"
    ) -> None:
        """Visualize the RGB image and its corresponding depth map side by side.
        
        Args:
            rgb_image: The original RGB image.
            depth_map: The estimated depth map.
            save_path: Optional path to save the visualization. If None, won't save.
            show: Whether to display the visualization using matplotlib.
            cmap: Colormap to use for the depth visualization.
        """
        plt.figure(figsize=(16, 8))
        
        # Original RGB image
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title("Original RGB Image")
        plt.axis("off")
        
        # Depth map
        plt.subplot(1, 2, 2)
        plt.imshow(depth_map, cmap=cmap)
        plt.title("Estimated Depth Map")
        plt.axis("off")
        
        plt.tight_layout()
        
        # Save visualization if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to: {save_path}")
        
        # Show visualization if requested
        if show:
            plt.show()
        
        plt.close()

# Convenient function for direct usage
def estimate_depth(
    img_path: Union[str, Path], 
    model_type: ModelType = "MiDaS_small",
    visualize: bool = False,
    save_visualization: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate depth from a single RGB image using MiDaS.
    
    This is a convenience function that creates a DepthEstimator instance
    and uses it to estimate depth from the given image.
    
    Args:
        img_path: Path to the input RGB image.
        model_type: MiDaS model variant to use.
        visualize: Whether to display the visualization.
        save_visualization: Path to save the visualization (if None, won't save).
        
    Returns:
        A tuple containing:
            - The original RGB image (numpy array)
            - The estimated depth map (numpy array)
    """
    estimator = DepthEstimator(model_type=model_type)
    rgb_image, depth_map = estimator.estimate_depth(img_path)
    
    if visualize or save_visualization:
        estimator.visualize_depth(
            rgb_image, 
            depth_map, 
            save_path=save_visualization,
            show=visualize
        )
    
    return rgb_image, depth_map 