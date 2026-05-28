"""
Utility functions for face recognition operations
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """Load an image from file path"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img


def convert_to_rgb(image):
    """Convert BGR image to RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image, width=None, height=None):
    """Resize image to specified dimensions"""
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None and height is None:
        ratio = width / w
        height = int(h * ratio)
    elif height is not None and width is None:
        ratio = height / h
        width = int(w * ratio)

    return cv2.resize(image, (width, height))


def save_image(image, output_path):
    """Save image to file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Image saved to {output_path}")


def get_image_info(image):
    """Get information about an image"""
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(image.dtype)
    }


def normalize_image(image):
    """Normalize image pixel values to 0-1 range"""
    return image.astype(np.float32) / 255.0


if __name__ == "__main__":
    print("Face recognition utilities loaded successfully")

