#!/usr/bin/env python
"""
Visualize referring expressions on images.

This script loads images and the referring expressions dataset, then displays
the images with bounding boxes around the objects that match each referring expression.

Usage:
    python visualize_referring_expressions.py --dataset path/to/refer_exp_dataset.jsonl --image_dir path/to/images
"""

import os
import json
import argparse
import random
from typing import Dict, List, Union, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Import BFSExpressionHandler for matching
from bfs_expression_handler import BFSExpressionHandler

# Define color mapping for different expression types
COLOR_MAP = {
    'DFS': 'blue',
    'BFS': 'red'
}

def load_dataset(dataset_path: str, bfs_only: bool = False) -> List[Dict]:
    """
    Load the referring expressions dataset.

    Args:
        dataset_path: Path to the referring expressions dataset
        bfs_only: Whether to filter for only BFS expressions

    Returns:
        List of dataset entries
    """
    with open(dataset_path, 'r') as f:
        entries = [json.loads(line) for line in f]

    # Filter for BFS expressions if requested
    if bfs_only:
        entries = [entry for entry in entries if entry['expression_type'] == 'BFS']
        print(f"Filtered dataset to {len(entries)} BFS expressions")

    return entries

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image as a numpy array.

    Args:
        image_path: Path to the image

    Returns:
        Image as a numpy array
    """
    try:
        return np.array(Image.open(image_path))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a placeholder image if the real image can't be loaded
        return np.zeros((600, 800, 3), dtype=np.uint8)

def visualize_referring_expression(entry: Dict, image_dir: str, ax=None, show: bool = True,
                                 save_path: str = None, no_assets: bool = False,
                                 bfs_handler: Optional[BFSExpressionHandler] = None) -> None:
    """
    Visualize a referring expression entry on its corresponding image.

    Args:
        entry: Referring expression dataset entry
        image_dir: Directory containing images
        ax: Matplotlib axes to plot on (optional)
        show: Whether to show the plot (default: True)
        save_path: Path to save the visualization (default: None)
        no_assets: Whether to disable visual assets (default: False)
        bfs_handler: BFSExpressionHandler for improved BFS matching (optional)
    """
    # Get the image path
    relative_image_path = entry['image_path']
    image_path = os.path.join(image_dir, os.path.basename(relative_image_path))

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found. Using a placeholder.")

    # Load the image
    image = load_image(image_path)

    # Create the figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Display the image
    ax.imshow(image)

    # Get referring expression
    expression = entry['referring_expression']
    expression_type = entry['expression_type']
    requirements = entry['target_requirements']

    # Get the color for this expression type
    color = COLOR_MAP.get(expression_type, 'green')

    # Set the title
    ax.set_title(f"{expression}", fontsize=12)

    # Add subtitle with expression type and requirements
    ax_subtitle = f"Type: {expression_type}, Requirements: "
    if expression_type == 'DFS':
        ax_subtitle += f"row={requirements['row']}, column={requirements['column']}"
    else:  # BFS
        # Handle both list and dictionary formats for requirements
        if isinstance(requirements, list):
            # For lists, join all requirements
            req_strs = []
            for req_dict in requirements:
                req_strs.append(", ".join(f"{k}={v}" for k, v in req_dict.items()))
            ax_subtitle += " | ".join(req_strs)
        else:
            # For dictionaries (older format)
            ax_subtitle += ", ".join(f"{k}={v}" for k, v in requirements.items())

    # Add reasoning level information if available
    if 'reasoning_level' in entry:
        ax_subtitle += f"\nReasoning Level: {entry['reasoning_level']}"

    # Add category information if available
    if 'category' in entry and 'category_description' in entry:
        ax_subtitle += f"\nCategory: {entry['category']} - {entry['category_description']}"

    ax.text(0.5, -0.05, ax_subtitle, transform=ax.transAxes, fontsize=10, ha='center')

    # Draw bounding boxes for matching objects
    for i, obj in enumerate(entry['matching_objects']):
        # Skip if we don't have bbox coordinates
        if 'bbox' not in obj:
            continue

        bbox = obj['bbox']

        # Handle different bbox formats (list of 4 values or [x1, y1, x2, y2])
        if len(bbox) == 4:
            if isinstance(bbox[0], (int, float)) and isinstance(bbox[2], (int, float)):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
            else:
                # Assume it's [left, top, width, height]
                x1, y1, width, height = bbox
        else:
            print(f"Warning: Unexpected bbox format: {bbox}")
            continue

        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add object index to the bounding box
        ax.text(
            x1,
            y1 - 5,
            f"Obj {i+1}",
            color=color,
            fontsize=8,
            weight='bold'
        )

    # Add object details as text (unless no_assets is True)
    if not no_assets:
        max_objects = min(3, len(entry['matching_objects']))  # Limit to 3 objects max
        for i, obj in enumerate(entry['matching_objects'][:max_objects]):
            # Skip if object doesn't have required attributes
            if not all(attr in obj for attr in ['shape_type', 'size', 'style', 'color1']):
                continue

            obj_desc = f"Obj {i+1}: {obj['shape_type']}, {obj['size']}, {obj['style']}, "
            obj_desc += f"{obj['color1']}"
            if obj['style'] in ['half', 'border'] and 'color2' in obj and obj['color1'] != obj['color2']:
                obj_desc += f"/{obj['color2']}"

            ax.text(
                0.02,
                0.98 - 0.05 * i,
                obj_desc,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7)
            )

    # Show or save the plot
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)

    # Remove axis ticks and borders
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    elif show:
        plt.show()

def visualize_dataset_sample(dataset_path: str, image_dir: str, num_samples: int = 5,
                           save_dir: str = None, bfs_only: bool = False,
                           no_assets: bool = False) -> None:
    """
    Visualize a random sample of referring expressions from the dataset.

    Args:
        dataset_path: Path to the referring expressions dataset
        image_dir: Directory containing images
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations (default: None)
        bfs_only: Whether to filter for only BFS expressions
        no_assets: Whether to disable visual assets (default: False)
    """
    # Load the dataset
    entries = load_dataset(dataset_path, bfs_only)

    if not entries:
        print(f"No {'BFS' if bfs_only else ''} entries found in the dataset.")
        return

    # Initialize BFSExpressionHandler for improved matching
    bfs_handler = BFSExpressionHandler()

    # Sample entries
    if len(entries) > num_samples:
        samples = random.sample(entries, num_samples)
    else:
        samples = entries

    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Visualize each sample individually
    for i, entry in enumerate(samples):
        # Create a new figure for each sample
        fig, ax = plt.subplots(figsize=(10, 8))

        # Remove axis ticks and borders
        ax.axis('off')

        # Generate save path if needed
        save_path = None
        if save_dir:
            expression_type = entry['expression_type']
            save_path = os.path.join(save_dir, f"{expression_type}_visualization_{i+1}.png")

        # Visualize the sample
        visualize_referring_expression(
            entry, image_dir, ax, show=False, save_path=save_path,
            no_assets=no_assets, bfs_handler=bfs_handler
        )
        plt.close(fig)

    # Create a combined visualization if we have multiple samples
    if len(samples) > 1 and save_dir:
        # Create a figure with subplots
        num_rows = (min(num_samples, len(samples)) + 1) // 2
        # Increase figure size to accommodate content better
        fig, axes = plt.subplots(num_rows, 2, figsize=(20, 8 * num_rows))
        if num_rows == 1:
            axes = [axes]

        # Flatten axes for easy indexing
        axes = axes.flatten()

        # Plot each sample
        for i, entry in enumerate(samples):
            if i < len(axes):
                visualize_referring_expression(
                    entry, image_dir, axes[i], show=False,
                    no_assets=no_assets, bfs_handler=bfs_handler
                )
                # Remove axis ticks and borders
                axes[i].axis('off')

        # Hide any unused subplots
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')

        # Add dataset info as a title
        plt.suptitle(f"Sample {'BFS' if bfs_only else ''} referring expressions from {os.path.basename(dataset_path)}", fontsize=14)

        # Use less restrictive layout parameters
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.2, hspace=0.4)

        # Save the combined visualization
        prefix = "BFS_" if bfs_only else ""
        combined_path = os.path.join(save_dir, f"{prefix}visualization_combined.png")
        plt.savefig(combined_path, dpi=100, bbox_inches='tight')
        print(f"Saved combined visualization to {combined_path}")
        plt.close(fig)

def generate_test_dataset(output_dir: str, num_images: int = 3, grid_size: Tuple[int, int] = (3, 3), sampling_strategy: str = "mixed") -> None:
    """
    Generate a small test dataset for visualization.

    Args:
        output_dir: Directory to save the test dataset
        num_images: Number of test images to generate
        grid_size: Grid size for the test images
        sampling_strategy: Strategy for generating expressions ("existence", "random", or "mixed")
    """
    import subprocess

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # Generate sample images
    print(f"Generating {num_images} test images with grid size {grid_size}...")
    cmd = [
        "python", "generate_samples.py",
        "--num-samples", str(num_images),
        "--output-dir", output_dir,
        "--min-grid", str(grid_size[0]),
        "--max-grid", str(grid_size[0]),
        "--seed", "42"
    ]
    subprocess.run(cmd)

    # Generate referring expressions
    print("Generating referring expressions dataset...")
    cmd = [
        "python", "create_referring_expressions_dataset.py",
        "--input", os.path.join(output_dir, "annotations", "dataset.jsonl"),
        "--output", os.path.join(output_dir, "annotations", "refer_exp_dataset.jsonl"),
        "--dfs_ratio", "0.5",
        "--sampling_strategy", sampling_strategy
    ]

    subprocess.run(cmd)

    print(f"Test dataset generated in {output_dir}")
    return os.path.join(output_dir, "annotations", "refer_exp_dataset.jsonl")

def main():
    """Parse command line arguments and run the visualization."""
    parser = argparse.ArgumentParser(description="Visualize referring expressions on images")
    parser.add_argument("--dataset", type=str, help="Path to the referring expressions dataset")
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--generate_test", action="store_true", help="Generate a test dataset")
    parser.add_argument("--test_dir", type=str, default="test_dataset", help="Directory for test dataset")
    parser.add_argument("--save_dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--bfs_only", action="store_true", help="Only visualize BFS (attribute-based) expressions")
    parser.add_argument("--no_assets", action="store_true", help="Disable visual assets in visualizations")
    parser.add_argument("--sampling_strategy", type=str, default="mixed", choices=["existence", "random", "mixed"],
                        help="Strategy for generating expressions (existence, random, or mixed)")

    args = parser.parse_args()

    # Generate a test dataset if requested
    if args.generate_test:
        dataset_path = generate_test_dataset(
            args.test_dir,
            sampling_strategy=args.sampling_strategy
        )
        image_dir = os.path.join(args.test_dir, "images")

        # Default save directory if not specified
        if not args.save_dir:
            args.save_dir = os.path.join(args.test_dir, "visualizations")
    else:
        # Use provided dataset and image directory
        dataset_path = args.dataset
        image_dir = args.image_dir

    # Check inputs
    if not dataset_path or not os.path.exists(dataset_path):
        print("Error: Dataset file not found.")
        return

    if not image_dir or not os.path.exists(image_dir):
        print("Error: Image directory not found.")
        return

    # Visualize the dataset
    visualize_dataset_sample(dataset_path, image_dir, args.num_samples, args.save_dir, args.bfs_only, args.no_assets)

if __name__ == "__main__":
    main()