#!/usr/bin/env python
"""Batch script for processing multiple images with the depth estimation module.

This script processes multiple images in a directory, generates depth maps,
and saves the results in specified output directories.
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from typing import List

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.depth_estimation.midas import DepthEstimator
from src.depth_estimation.utils import save_depth_map, export_depth_data, save_depth_visualization


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Batch depth estimation from RGB images using MiDaS.")
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory containing input RGB images."
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Base directory for output files. Defaults to 'results'."
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"],
        default="MiDaS_small",
        help="MiDaS model type to use. Defaults to MiDaS_small."
    )
    
    parser.add_argument(
        "--extensions", "-e",
        type=str,
        default=".jpg,.jpeg,.png",
        help="Comma-separated list of image file extensions to process. Defaults to .jpg,.jpeg,.png."
    )
    
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw depth data as .npy files."
    )
    
    parser.add_argument(
        "--save-colored",
        action="store_true",
        help="Save colored depth maps as images."
    )
    
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save side-by-side visualizations of RGB and depth."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for computation. If not specified, will use CUDA if available."
    )
    
    return parser.parse_args()


def find_images(input_dir: str, extensions: List[str]) -> List[str]:
    """Find all images with specified extensions in the input directory.
    
    Args:
        input_dir: Directory to search for images.
        extensions: List of file extensions to include.
        
    Returns:
        List of paths to image files.
    """
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        image_files.extend(glob.glob(pattern))
    
    return sorted(image_files)


def main() -> None:
    """Run batch depth estimation on the provided images."""
    args = parse_args()
    
    # Parse extensions
    extensions = args.extensions.split(",")
    
    # Find images
    image_files = find_images(args.input_dir, extensions)
    if not image_files:
        print(f"Error: No images with extensions {extensions} found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    
    vis_dir = output_dir / "visualizations"
    depth_dir = output_dir / "depth_maps"
    raw_dir = output_dir / "raw_depth"
    
    if args.save_vis:
        vis_dir.mkdir(exist_ok=True, parents=True)
    
    if args.save_colored:
        depth_dir.mkdir(exist_ok=True, parents=True)
    
    if args.save_raw:
        raw_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize depth estimator
    device = None
    if args.device:
        import torch
        device = torch.device(args.device)
    
    try:
        estimator = DepthEstimator(model_type=args.model, device=device)
        
        # Process each image
        for i, img_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {img_path}")
            
            try:
                # Get image filename without extension
                img_filename = Path(img_path).stem
                
                # Estimate depth
                rgb_image, depth_map = estimator.estimate_depth(img_path)
                
                # Save outputs according to options
                if args.save_vis:
                    vis_path = vis_dir / f"{img_filename}_depth_vis.png"
                    save_depth_visualization(rgb_image, depth_map, vis_path)
                
                if args.save_colored:
                    depth_path = depth_dir / f"{img_filename}_depth.png"
                    save_depth_map(depth_map, depth_path)
                
                if args.save_raw:
                    raw_path = raw_dir / f"{img_filename}_depth.npy"
                    export_depth_data(depth_map, raw_path)
                
                print(f"  - Processed successfully")
                
            except Exception as e:
                print(f"  - Error processing {img_path}: {e}")
        
        print("Batch processing complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 