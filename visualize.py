#!/usr/bin/env python
"""
Script to visualize an image with its annotations.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize an image with its annotations')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--annotation', type=str,
                        help='Path to the annotation file (if not provided, will look for a .json file with the same name)')
    parser.add_argument('--jsonl', type=str,
                        help='Path to JSONL file containing annotations for multiple images')
    parser.add_argument('--show-grid', action='store_true',
                        help='Show grid positions')
    parser.add_argument('--show-bbox', action='store_true',
                        help='Show bounding boxes')
    parser.add_argument('--show-region', action='store_true',
                        help='Show the region boundaries')
    parser.add_argument('--output', type=str,
                        help='Path to save the visualization (if not provided, will display the image)')
    return parser.parse_args()

def load_from_jsonl(jsonl_path, image_path):
    """Find the annotation for an image in a JSONL file."""
    image_name = os.path.basename(image_path)
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if os.path.basename(data["image_path"]) == image_name:
                return data
    return None

def visualize_annotations(image_path, annotation_path=None, jsonl_path=None,
                         show_grid=False, show_bbox=False, show_region=False, output_path=None):
    """
    Visualize an image with its annotations.
    """
    # Load image
    img = Image.open(image_path)

    # Find annotation data
    data = None

    # Try JSONL first if provided
    if jsonl_path:
        data = load_from_jsonl(jsonl_path, image_path)
        if data:
            print(f"Found annotation in JSONL file for {os.path.basename(image_path)}")

    # Fall back to direct JSON file
    if data is None:
        # Find annotation file if not provided
        if annotation_path is None:
            base_path = os.path.splitext(image_path)[0]
            annotation_path = f"{base_path}.json"

        # Load annotations
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Could not find annotation at {annotation_path}")
            if jsonl_path:
                print(f"Also could not find in JSONL: {jsonl_path}")
            return

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(np.array(img))

    # Add title
    ax.set_title(f"Image: {os.path.basename(image_path)}")

    # Display region if requested
    if show_region and "region" in data["scene"]:
        region_x, region_y, region_width, region_height = data["scene"]["region"]
        region_rect = patches.Rectangle(
            (region_x, region_y), region_width, region_height,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
        )
        ax.add_patch(region_rect)
        ax.text(region_x + 5, region_y + region_height - 10, f"Region: {region_width}x{region_height}",
                color='blue', fontsize=8, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))

    # Display shapes information
    shapes = data['scene']['shapes']

    # Create colors for visualization
    colors = {
        'red': 'red',
        'green': 'green',
        'blue': 'blue',
        'yellow': 'yellow',
        'purple': 'purple',
        'orange': 'orange'
    }

    # Add annotations
    for i, shape in enumerate(shapes):
        # Get shape information
        shape_type = shape['shape_type']
        position = shape['position']
        bbox = shape['bbox']
        grid_pos = shape['grid_position']
        color = colors.get(shape['color1'], 'white')

        # Add bounding box
        if show_bbox:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Add grid position
        if show_grid:
            ax.text(position[0], position[1], f"({grid_pos[0]},{grid_pos[1]})",
                    color='white', fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7))

        # Add shape label
        label = f"{i+1}: {shape_type} ({shape['size']})"
        ax.text(position[0], position[1] - 15, label,
                color='white', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5))

    # Remove axes
    ax.set_axis_off()

    # Save or show the figure
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()

def main():
    args = parse_args()
    visualize_annotations(
        args.image,
        args.annotation,
        args.jsonl,
        args.show_grid,
        args.show_bbox,
        args.show_region,
        args.output
    )

if __name__ == "__main__":
    main()