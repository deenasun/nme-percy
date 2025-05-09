#!/usr/bin/env python
"""Script to test the depth estimation module.

This script takes an input image and generates a depth map using the MiDaS model.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.depth_estimation.midas import estimate_depth


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Estimate depth from an RGB image using MiDaS.")
    
    parser.add_argument(
        "--image", "-i", 
        type=str, 
        required=True, 
        help="Path to the input RGB image."
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"], 
        default="MiDaS_small",
        help="MiDaS model type to use. Defaults to MiDaS_small."
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        help="Path to save the visualization. If not provided, will use the input filename with '_depth' suffix."
    )
    
    parser.add_argument(
        "--no-visualize", 
        action="store_true", 
        help="Don't display the visualization (just save it)."
    )
    
    return parser.parse_args()


def main() -> None:
    """Run the depth estimation on the provided image."""
    args = parse_args()
    
    # Validate input path
    if not os.path.isfile(args.image):
        print(f"Error: Input image file not found: {args.image}")
        sys.exit(1)
    
    # Determine output path if not provided
    if args.output is None:
        input_path = Path(args.image)
        output_dir = Path("results/depth_maps")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{input_path.stem}_depth.png"
    else:
        output_path = args.output
    
    try:
        # Estimate depth
        print(f"Estimating depth for image: {args.image}")
        print(f"Using model: {args.model}")
        
        # Run depth estimation
        estimate_depth(
            img_path=args.image,
            model_type=args.model,
            visualize=not args.no_visualize,
            save_visualization=output_path
        )
        
        print(f"Depth visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 