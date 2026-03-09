import os
from typing import Dict, Any, Tuple
import cv2
import numpy as np


def parse_annotation_line(line: str) -> Dict[str, Any]:
    """
	Parse a single WFLW annotation line.

	Args:
		line (str): Raw annotation line from a WFLW annotation file.

    Returns:
        Dict[str, Any]: Dictionary containing:
            landmarks (np.ndarray): Array of shape (98, 2) containing
                    facial landmark coordinates.
                bbox (list[float]): Bounding box coordinates
                    [x_min, y_min, x_max, y_max].
                attributes (dict): Metadata labels.
                image_relative_path (str): Relative path to the image file.
    """
    parts = line.strip().split(",")
    if len(parts) < 207:
        raise ValueError(f"Invalid annotation line: {line}")
    landmark_vals = np.array(list(map(float, parts[:196])), dtype=np.float32)
    landmarks = landmark_vals.reshape(98, 2)
    bbox = list(map(float, parts[196:200]))
    attributes = {
        "pose": int(parts[200]),
        "expression": int(parts[201]),
        "illumination": int(parts[202]),
        "makeup": int(parts[203]),
        "occlusion": int(parts[204]),
        "blur": int(parts[205]),
    }
    image_relative_path = parts[206]
    return {
        "landmarks": landmarks,
        "bbox": bbox,
        "attributes": attributes,
        "image_relative_path": image_relative_path,
    }


def load_annotation_by_idx(idx: int, annotation_file: str) -> Dict[str, Any]:
    """
	Load a WFLW annotation by index.

	Args:
		annotation_file (str): Path to the WFLW annotation file.
		index (int): Line index of the sample to load.

    Returns:
        Dict[str, Any]: Parsed annotation dictionary.
    """
    with open(annotation_file, "r") as f:
        lines = f.readlines()
    if idx < 0 or idx >= len(lines):
        raise IndexError(f"Index {idx} out of bounds")
    return parse_annotation_line(lines[idx])


def load_image(image_dir: str, relative_path: str) -> Tuple[np.ndarray, str]:
    """
    Load an image given its relative path.

    Args:
		image_dir (str): Directory for WFLW image.
		relative_path (str): Relative path to the image file.

    Returns:
        Tuple[np.ndarray, str]:
			image (np.ndarray): Retrieved image.
			image_path (str): Full path to the image.
    """
    image_path = os.path.join(image_dir, relative_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image, image_path


def draw_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    """
    Draw facial landmarks on an image.

    Args:
		image (np.ndarray): Input image.
		landmarks (np.ndarray): Array of shape (98, 2) with landmark coordinates.
		color (Tuple[int, int, int]): Color to draw the landmarks.
		radius (int): Radius of the landmark points.

    Returns:
        np.ndarray: Image with landmarks drawn.
    """
    for x, y in landmarks.astype(int):
        cv2.circle(image, (x, y), radius, color, -1)
    return image


def draw_bbox(
    image: np.ndarray,
    bbox: list,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box on an image.

    Args:
		image (np.ndarray): Input image.
		bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max].
		color (Tuple[int, int, int]): Color to draw the bounding box.
		thickness (int): Thickness of the bounding box lines.

    Returns:
        np.ndarray: Image with bounding box drawn.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image
