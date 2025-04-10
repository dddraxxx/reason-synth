import os
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from .shapes import Shape, COLORS, get_random_attributes
from .scene import Scene

def generate_random_shape() -> Shape:
    """Generate a random shape with random attributes using the configuration."""
    # Get random attributes from config
    attrs = get_random_attributes()

    return Shape(
        shape_type=attrs["shape_type"],
        size=attrs["size"],
        color1=attrs["color1"],
        color2=attrs["color2"],
        style=attrs["style"],
        rotation=attrs["rotation"]
    )

def generate_dataset(
    num_samples: int = 100,
    grid_sizes: List[Tuple[int, int]] = [(2, 2), (2, 3), (3, 2), (3, 3)],
    output_dir: str = "data",
    image_subdir: str = "images",
    annotation_file: str = "annotations/dataset.jsonl",
    cell_size: int = 70,
    region_variation_ratio: float = 0.1,
    max_offset_ratio: float = 0.2,
    max_overlap_ratio: float = 0.1,
    seed: Optional[int] = None
) -> None:
    """
    Generate a dataset of synthetic images with annotations.

    Args:
        num_samples: Number of images to generate
        grid_sizes: List of possible grid sizes (rows, columns)
        output_dir: Base output directory
        image_subdir: Subdirectory for images
        annotation_file: Path to annotation file (relative to output_dir)
        cell_size: Base size of a grid cell in pixels
        region_variation_ratio: Random variation (+/-) ratio for placement region size
        max_offset_ratio: Maximum offset as a ratio of cell size
        max_overlap_ratio: Max allowed overlap distance ratio (relative to smaller bbox diagonal)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create output directories
    image_dir = os.path.join(output_dir, image_subdir)
    os.makedirs(image_dir, exist_ok=True)

    # Make sure annotations directory exists
    annotation_dir = os.path.dirname(os.path.join(output_dir, annotation_file))
    os.makedirs(annotation_dir, exist_ok=True)

    # Full path to annotation file
    annotation_path = os.path.join(output_dir, annotation_file)

    print(f"Image directory: {image_dir}")
    print(f"Annotation path: {annotation_path}")

    # Delete existing annotation file if it exists
    if os.path.exists(annotation_path):
        os.remove(annotation_path)
        print(f"Removed existing annotation file: {annotation_path}")

    # Initialize statistics tracking
    total_stats = {
        "total_shapes": 0,
        "successful_placements": 0,
        "failed_placements": 0,
        "total_attempts": 0
    }

    # Generate samples
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Choose a random grid size
        grid_size = random.choice(grid_sizes)
        rows, cols = grid_size

        # Create a scene with random region
        scene = Scene(
            grid_size=grid_size,
            cell_size=cell_size,
            region_variation_ratio=region_variation_ratio,
            max_offset_ratio=max_offset_ratio,
            max_overlap_ratio=max_overlap_ratio
        )

        # Add shapes to the scene
        for row in range(rows):
            for col in range(cols):
                shape = generate_random_shape()
                scene.add_shape(shape, (row, col))

        # Save image
        image_filename = f"sample_{i:04d}.png"
        image_path = os.path.join(image_dir, image_filename)
        scene.save_image(image_path)
        print(f"Saved image {i+1}/{num_samples}: {image_path}")

        # Save annotation
        relative_image_path = os.path.join(image_subdir, image_filename)
        scene.save_to_jsonl(annotation_path, relative_image_path)

        # Collect stats from this scene
        stats = scene.get_placement_stats()
        total_stats["total_shapes"] += stats["total_shapes"]
        total_stats["successful_placements"] += stats["successful_placements"]
        total_stats["failed_placements"] += stats["failed_placements"]
        total_stats["total_attempts"] += stats["total_attempts"]

    # Calculate overall statistics
    if total_stats["total_shapes"] > 0:
        success_rate = (total_stats["successful_placements"] / total_stats["total_shapes"]) * 100
        avg_attempts = total_stats["total_attempts"] / total_stats["total_shapes"]
    else:
        success_rate = 0.0
        avg_attempts = 0.0

    # Print statistics report
    print(f"Generated {num_samples} samples in {image_dir}")
    print(f"Annotations saved to {annotation_path}")
    print("\nShape Placement Statistics:")
    print(f"  Total shapes placed: {total_stats['total_shapes']}")
    print(f"  Successful placements: {total_stats['successful_placements']} ({success_rate:.2f}%)")
    print(f"  Failed placements: {total_stats['failed_placements']}")
    print(f"  Average placement attempts per shape: {avg_attempts:.2f}")

def generate_sample_image(output_path: str = "sample.png") -> None:
    """Generate a single sample image for testing."""
    # Create a 2x3 grid with random region
    scene = Scene(
        grid_size=(2, 3),
        image_size=(600, 400),
        cell_size=70,
        region_variation_ratio=0.1,
        max_offset_ratio=0.2,
        max_overlap_ratio=0.1
    )

    # Add different shapes
    shapes = [
        Shape("triangle", "big", "red", style="solid", rotation=30),
        Shape("square", "small", "blue", style="solid"),
        Shape("circle", "big", "green", style="solid"),
        Shape("triangle", "small", "purple", "yellow", style="half"),
        Shape("square", "big", "orange", "blue", style="border"),
        Shape("circle", "small", "yellow", "red", style="half")
    ]

    # Add shapes to grid positions
    grid_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for shape, pos in zip(shapes, grid_positions):
        scene.add_shape(shape, pos)

    # Save image
    scene.save_image(output_path)

    # Save annotation
    annotation_path = os.path.splitext(output_path)[0] + ".json"
    scene.save_annotation(annotation_path, output_path)

    print(f"Sample image saved to {output_path}")
    print(f"Sample annotation saved to {annotation_path}")

    # Report placement statistics
    stats = scene.get_placement_stats()
    if stats["total_shapes"] > 0:
        success_rate = stats["success_rate"]
        avg_attempts = stats["avg_attempts_per_shape"]
    else:
        success_rate = 0.0
        avg_attempts = 0.0

    print("\nShape Placement Statistics:")
    print(f"  Total shapes placed: {stats['total_shapes']}")
    print(f"  Successful placements: {stats['successful_placements']} ({success_rate:.2f}%)")
    print(f"  Failed placements: {stats['failed_placements']}")
    print(f"  Average placement attempts per shape: {avg_attempts:.2f}")

if __name__ == "__main__":
    # Generate a sample image
    generate_sample_image()

    # Generate a small dataset
    generate_dataset(num_samples=10, seed=42)
