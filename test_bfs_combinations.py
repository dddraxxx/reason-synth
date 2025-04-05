"""
Test script for BFS expression combinations.

This script uses generate_samples.py to create a single image with a 6x6 grid of objects,
then generates and visualizes BFS expressions for these objects.
"""

import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import argparse
import numpy as np
import subprocess
from typing import Dict, List, Tuple, Any, Optional

# Import the BFSExpressionHandler
from bfs_expression_handler import BFSExpressionHandler, SHAPE_TYPES, COLORS, SIZES, STYLES

# Constants
OUTPUT_DIR = "bfs_test_combinations"
GRID_SIZE = (6, 6)

def generate_sample_data(output_dir: str) -> Tuple[List[Dict], str]:
    """
    Generate sample data using generate_samples.py.

    Args:
        output_dir: Output directory for the dataset

    Returns:
        Tuple of (objects list, image path)
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/annotations", exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)

    # Call generate_samples.py to create a single sample with a 6x6 grid
    # Don't use the --sample flag as it might not respect our grid size parameters
    cmd = [
        "python", "generate_samples.py",
        "--num-samples", "1",
        "--output-dir", output_dir,
        "--min-grid", "6",
        "--max-grid", "6",
        "--seed", "42"
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

    # Read the generated dataset
    dataset_path = f"{output_dir}/annotations/dataset.jsonl"

    # Load the dataset
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    if not dataset:
        print("Warning: Empty dataset found")
        return [], ""

    # Use the first image in the dataset
    image_data = dataset[0]

    # Check the dataset format
    if "scene" in image_data and "shapes" in image_data["scene"]:
        # New dataset format
        grid_size = image_data["scene"]["grid_size"]
        shapes = image_data["scene"]["shapes"]

        # Convert shapes to our expected format
        objects = []
        for shape in shapes:
            # Create a normalized bbox from the actual bbox
            # Our visualizations expect bbox in normalized coordinates [x1, y1, x2, y2] from 0 to 1
            img_width, img_height = image_data["scene"]["image_size"]
            x1, y1, x2, y2 = shape["bbox"]
            normalized_bbox = [
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height
            ]

            # Add the shape to our objects list
            objects.append({
                "shape_type": shape["shape_type"],
                "size": shape["size"],
                "style": shape["style"],
                "color1": shape["color1"],
                "color2": shape["color2"],
                "grid_position": shape["grid_position"],
                "bbox": normalized_bbox
            })

        # Use the image path from the dataset
        relative_image_path = image_data["image_path"]
        image_path = os.path.join(output_dir, relative_image_path)
    else:
        # Old dataset format
        objects = []
        if "objects" in image_data:
            for obj in image_data["objects"]:
                objects.append(obj)

        # Use the image path from the dataset
        relative_image_path = image_data["image_path"]
        image_path = os.path.join(output_dir, relative_image_path)

    return objects, image_path

def generate_bfs_combinations(objects: List[Dict]) -> List[Dict]:
    """
    Generate a comprehensive set of BFS expressions for testing.

    Args:
        objects: List of objects in the image

    Returns:
        List of referring expressions
    """
    # Create a BFSExpressionHandler instance
    handler = BFSExpressionHandler()

    # Generate expressions using the "patterns" strategy to get one example per pattern
    # Limit to a small subset of objects to reduce the number of expressions
    limited_objects = objects[:10]  # Just use the first 10 objects

    print(f"Using {len(limited_objects)} objects out of {len(objects)} total")

    expressions = handler.generate_bfs_expressions(limited_objects, sampling_strategy="patterns", samples_per_pattern=1)
    # shuffle and keep the first 30
    random.shuffle(expressions)
    expressions = expressions[:30]

    print(f"Generated {len(expressions)} expressions before matching")

    # Process the expressions to add matching_objects field for visualization
    for expr in expressions:
        expr["matching_objects"] = handler.find_matching_objects(expr["target_requirements"], objects)

    # Keep only expressions that have matching objects
    expressions = [expr for expr in expressions if expr["matching_objects"]]

    print(f"Kept {len(expressions)} expressions after filtering")

    return expressions

def visualize_expressions(image_path: str, expressions: List[Dict], output_dir: str):
    """
    Create visualizations of expressions with matching objects highlighted.

    Args:
        image_path: Path to the input image
        expressions: List of referring expressions
        output_dir: Directory to save visualizations
    """
    print(f"Starting visualization with {len(expressions)} expressions")
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)

    # Load the original image
    try:
        print(f"Loading image from {image_path}")
        original_img = plt.imread(image_path)
        img_height, img_width = original_img.shape[:2]
        print(f"Image loaded, size: {img_width}x{img_height}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Create a placeholder image
        img_width, img_height = 600, 600
        original_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        print("Created placeholder image")

    # Create a single combined visualization
    # Group expressions by pattern category to create a more organized visualization
    categories = {
        "single_attribute": ["shape", "color", "size", "style"],
        "two_attributes": ["shape_color", "shape_size", "shape_style", "color_size", "color_style", "size_style"],
        "three_attributes": ["shape_color_size", "shape_color_style", "shape_size_style", "color_size_style"],
        "all_attributes": ["shape_color_size_style"],
    }

    # Sort expressions by category
    categorized_expressions = {}
    for category, patterns in categories.items():
        categorized_expressions[category] = [expr for expr in expressions if expr["pattern_key"] in patterns]
        print(f"Category {category}: {len(categorized_expressions[category])} expressions")

    # For each category, create a visualization page with expressions in that category
    for category, category_expressions in categorized_expressions.items():
        if not category_expressions:
            print(f"Skipping empty category: {category}")
            continue

        try:
            # Calculate layout dimensions
            num_exprs = len(category_expressions)
            if num_exprs <= 3:
                rows, cols = num_exprs, 1
            else:
                cols = min(3, num_exprs)
                rows = (num_exprs + cols - 1) // cols

            print(f"Creating {rows}x{cols} grid for {category} with {num_exprs} expressions")

            # Create figure
            fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5), constrained_layout=True)

            # Handle single subplot case
            if rows * cols == 1:
                axes = np.array([axes])

            # Flatten axes for easy indexing
            axes = axes.flatten()

            # Plot each expression
            for i, expr in enumerate(category_expressions):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Show the image
                ax.imshow(original_img)

                # Get the expression
                expression = expr['referring_expression']
                pattern_key = expr['pattern_key']

                # Highlight matching objects
                for obj in expr["matching_objects"]:
                    # Get the normalized bbox coordinates [x1, y1, x2, y2]
                    norm_bbox = obj["bbox"]

                    # Convert normalized coordinates to image pixel coordinates
                    # Note: matplotlib's imshow has (0,0) at the top-left corner
                    bbox_pixels = [
                        norm_bbox[0] * img_width,            # x1
                        norm_bbox[1] * img_height,           # y1
                        (norm_bbox[2] - norm_bbox[0]) * img_width,  # width
                        (norm_bbox[3] - norm_bbox[1]) * img_height  # height
                    ]

                    # Draw bounding box
                    rect = patches.Rectangle(
                        (bbox_pixels[0], bbox_pixels[1]),  # (x, y)
                        bbox_pixels[2],  # width
                        bbox_pixels[3],  # height
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add grid position if available
                    if "grid_position" in obj:
                        grid_pos = obj["grid_position"]
                        center_x = bbox_pixels[0] + bbox_pixels[2] / 2
                        center_y = bbox_pixels[1] + bbox_pixels[3] / 2

                        ax.text(
                            center_x, center_y,
                            f"({grid_pos[0]},{grid_pos[1]})",
                            ha='center', va='center',
                            color='black', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1')
                        )

                # Set title with the expression and pattern
                ax.set_title(f"{pattern_key}: {expression}", fontsize=12)

                # Add requirements as subtitle
                requirements = expr['target_requirements']
                req_text = ""
                if isinstance(requirements, list) and len(requirements) > 0:
                    req_text = ", ".join(f"{k}={v}" for k, v in requirements[0].items())
                ax.text(
                    0.5, -0.05,
                    req_text,
                    transform=ax.transAxes,
                    fontsize=10, ha='center'
                )

                ax.set_xticks([])
                ax.set_yticks([])

            # Hide any unused subplots
            for i in range(num_exprs, len(axes)):
                axes[i].axis('off')

            # Add category as title
            plt.suptitle(f"Category: {category.replace('_', ' ').title()}", fontsize=16)

            # Save the visualization
            output_path = f"{output_dir}/visualizations/bfs_{category}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {category} visualization to {output_path}")
        except Exception as e:
            print(f"Error creating visualization for {category}: {e}")

    # Create a combined visualization of all expressions
    num_exprs = len(expressions)
    rows = (num_exprs + 2) // 3  # 3 columns
    cols = min(3, num_exprs)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5), constrained_layout=True)

    # Handle single subplot case
    if rows * cols == 1:
        axes = np.array([axes])

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Plot each expression
    for i, expr in enumerate(expressions):
        if i >= len(axes):
            break

        ax = axes[i]

        # Show the image
        ax.imshow(original_img)

        # Get the expression
        expression = expr['referring_expression']
        pattern_key = expr['pattern_key']

        # Highlight matching objects
        for obj in expr["matching_objects"]:
            # Get the normalized bbox coordinates [x1, y1, x2, y2]
            norm_bbox = obj["bbox"]

            # Convert normalized coordinates to image pixel coordinates
            # Note: matplotlib's imshow has (0,0) at the top-left corner
            bbox_pixels = [
                norm_bbox[0] * img_width,            # x1
                norm_bbox[1] * img_height,           # y1
                (norm_bbox[2] - norm_bbox[0]) * img_width,  # width
                (norm_bbox[3] - norm_bbox[1]) * img_height  # height
            ]

            # Draw bounding box
            rect = patches.Rectangle(
                (bbox_pixels[0], bbox_pixels[1]),  # (x, y)
                bbox_pixels[2],  # width
                bbox_pixels[3],  # height
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add grid position if available
            if "grid_position" in obj:
                grid_pos = obj["grid_position"]
                center_x = bbox_pixels[0] + bbox_pixels[2] / 2
                center_y = bbox_pixels[1] + bbox_pixels[3] / 2

                ax.text(
                    center_x, center_y,
                    f"({grid_pos[0]},{grid_pos[1]})",
                    ha='center', va='center',
                    color='black', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1')
                )

        # Set title with the expression and pattern
        ax.set_title(f"{pattern_key}: {expression}", fontsize=12)

        # Add requirements as subtitle
        requirements = expr['target_requirements']
        req_text = ""
        if isinstance(requirements, list) and len(requirements) > 0:
            req_text = ", ".join(f"{k}={v}" for k, v in requirements[0].items())
        ax.text(
            0.5, -0.05,
            req_text,
            transform=ax.transAxes,
            fontsize=10, ha='center'
        )

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for i in range(num_exprs, len(axes)):
        axes[i].axis('off')

    # Add title
    plt.suptitle(f"All BFS Expression Patterns ({num_exprs} total)", fontsize=16)

    # Save the visualization
    output_path = f"{output_dir}/visualizations/bfs_all_patterns.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined visualization to {output_path}")

def main():
    """
    Main function to run the test script.
    """
    parser = argparse.ArgumentParser(description="Test BFS expression combinations")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for test data")
    args = parser.parse_args()

    # Generate sample data
    print("Generating sample data...")
    objects, image_path = generate_sample_data(args.output_dir)

    # Generate BFS expression combinations
    print("Generating BFS expression combinations...")
    expressions = generate_bfs_combinations(objects)

    # Save expressions to a file
    os.makedirs(f"{args.output_dir}/annotations", exist_ok=True)
    with open(f"{args.output_dir}/annotations/bfs_expressions.jsonl", "w") as f:
        for expr in expressions:
            # Convert matching_objects to serializable format for JSON
            serializable_expr = expr.copy()
            serializable_expr["matching_objects"] = [
                {"grid_position": obj["grid_position"]} for obj in expr["matching_objects"]
            ]
            f.write(json.dumps(serializable_expr) + "\n")

    print(f"Generated {len(expressions)} BFS expressions")

    # Visualize expressions with matching objects
    visualize_expressions(image_path, expressions, args.output_dir)

    print(f"Test image saved to {image_path}")
    print(f"All visualizations saved to {args.output_dir}/visualizations/")

if __name__ == "__main__":
    main()