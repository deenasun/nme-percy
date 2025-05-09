#!/usr/bin/env python
"""Download a sample image for testing depth estimation.

This script downloads a sample image from the MiDaS repository
and saves it to the data/images directory.
"""

import os
import sys
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def download_sample_image() -> Path:
    """Download a sample image from the MiDaS repository.
    
    Returns:
        Path to the downloaded image.
    """
    # URL of a sample image
    sample_img_url = "https://github.com/intel-isl/MiDaS/raw/master/utils/img/pexels-photo-4200563.jpeg"
    
    # Directory to save the image
    image_dir = project_root / "data" / "images"
    os.makedirs(image_dir, exist_ok=True)
    
    # Path to save the image
    sample_img_path = image_dir / "sample.jpg"
    
    # Download the image
    try:
        print(f"Downloading sample image from {sample_img_url}")
        response = requests.get(sample_img_url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Load the image
        img = Image.open(BytesIO(response.content))
        
        # Save the image
        img.save(sample_img_path)
        
        print(f"Downloaded sample image to {sample_img_path}")
        return sample_img_path
    
    except Exception as e:
        print(f"Error downloading sample image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_sample_image() 