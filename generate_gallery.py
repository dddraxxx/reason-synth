#!/usr/bin/env python
"""
Script to generate a gallery image showing all shape, size, and style combinations.
"""

import os
import numpy as np
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from src.shapes import Shape, COLORS
from src.scene import Scene

def generate_gallery(output_path="gallery.png", rotated_output_path=None):
    """Generate a gallery image showing all shape, size, and style combinations."""
    # Define all values for the gallery
    shapes = ["triangle", "square", "circle"]
    sizes = ["big", "small"]
    styles = ["solid", "half", "border"]
    colors = list(COLORS.keys())

    # Image parameters
    cell_size = 120
    padding = 20

    # Select a consistent color combination
    color1 = "blue"
    color2 = "red"

    # Calculate dimensions
    num_shapes = len(shapes)
    num_sizes = len(sizes)
    num_styles = len(styles)

    # Calculate the number of columns and rows
    cols = num_shapes
    rows = num_sizes * num_styles

    # Create the image
    width = cols * cell_size + (cols + 1) * padding
    height = rows * cell_size + (rows + 1) * padding
    img = Image.new('RGBA', (width, height), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)

    # Draw grid lines
    for i in range(cols + 1):
        x = i * cell_size + (i + 0.5) * padding
        draw.line([(x, 0), (x, height)], fill=(200, 200, 200), width=1)

    for i in range(rows + 1):
        y = i * cell_size + (i + 0.5) * padding
        draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)

    # Draw header labels for shapes
    for i, shape in enumerate(shapes):
        x = i * cell_size + (i + 1) * padding + cell_size // 2
        y = padding // 2
        draw.text((x, y), shape.capitalize(), fill=(0, 0, 0), anchor="mm")

    # Draw all shapes
    for i, shape_type in enumerate(shapes):
        for j, size in enumerate(sizes):
            for k, style in enumerate(styles):
                # Calculate position in the grid
                row = j * num_styles + k
                col = i

                # Calculate center of the cell
                cx = col * cell_size + (col + 1) * padding + cell_size // 2
                cy = row * cell_size + (row + 1) * padding + cell_size // 2

                # Create and draw the shape
                shape = Shape(
                    shape_type=shape_type,
                    size=size,
                    color1=color1,
                    color2=color2 if style in ["half", "border"] else color1,
                    style=style,
                    position=(cx, cy),
                    rotation=0
                )
                shape.draw(img)

                # Add label for size and style
                label = f"{size}, {style}"
                draw.text((cx, cy + 40), label, fill=(0, 0, 0), anchor="mm")

    # Save the image
    img.save(output_path)
    print(f"Gallery image saved to {output_path}")

    # Set default rotated output path if not provided
    if rotated_output_path is None:
        rotated_output_path = os.path.splitext(output_path)[0] + "_rotated.png"

    # Create a rotated version
    img_rotated = Image.new('RGBA', (width, height), (240, 240, 240, 255))
    draw_rotated = ImageDraw.Draw(img_rotated)

    # Draw grid lines
    for i in range(cols + 1):
        x = i * cell_size + (i + 0.5) * padding
        draw_rotated.line([(x, 0), (x, height)], fill=(200, 200, 200), width=1)

    for i in range(rows + 1):
        y = i * cell_size + (i + 0.5) * padding
        draw_rotated.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)

    # Draw header labels for shapes
    for i, shape in enumerate(shapes):
        x = i * cell_size + (i + 1) * padding + cell_size // 2
        y = padding // 2
        draw_rotated.text((x, y), shape.capitalize(), fill=(0, 0, 0), anchor="mm")

    # Draw all shapes with rotation
    for i, shape_type in enumerate(shapes):
        for j, size in enumerate(sizes):
            for k, style in enumerate(styles):
                # Calculate position in the grid
                row = j * num_styles + k
                col = i

                # Calculate center of the cell
                cx = col * cell_size + (col + 1) * padding + cell_size // 2
                cy = row * cell_size + (row + 1) * padding + cell_size // 2

                # Skip rotation for circles
                rotation = 45.0 if shape_type != "circle" else 0.0

                # Create and draw the shape
                shape = Shape(
                    shape_type=shape_type,
                    size=size,
                    color1=color1,
                    color2=color2 if style in ["half", "border"] else color1,
                    style=style,
                    position=(cx, cy),
                    rotation=rotation
                )
                shape.draw(img_rotated)

                # Add label for size and style
                label = f"{size}, {style}"
                draw_rotated.text((cx, cy + 40), label, fill=(0, 0, 0), anchor="mm")

    # Save the rotated image
    img_rotated.save(rotated_output_path)
    print(f"Rotated gallery image saved to {rotated_output_path}")

    return output_path, rotated_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a gallery of all shape combinations")
    parser.add_argument("--output", default="gallery.png", help="Output path for the gallery image")
    parser.add_argument("--rotated-output", default=None, help="Output path for the rotated gallery image")
    args = parser.parse_args()

    generate_gallery(args.output, args.rotated_output)