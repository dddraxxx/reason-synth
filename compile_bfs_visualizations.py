#!/usr/bin/env python
"""
Compile all BFS (attribute-based) visualizations into a single HTML report.

This script scans all the visualization directories, collects the BFS visualization
images, and creates an HTML report with organized sections for different types of
attribute-based referring expressions.
"""

import os
import glob
import argparse
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

# Define the different example categories
EXAMPLE_CATEGORIES = [
    {
        "dir": "visualization_test2/bfs_visualizations",
        "title": "General BFS Examples",
        "description": "Various attribute-based referring expressions from the test dataset."
    },
    {
        "dir": "bfs_examples/visualizations",
        "title": "Red Triangle with Border",
        "description": "BFS expressions targeting triangles with a red color and border style."
    },
    {
        "dir": "bfs_examples_triangles/visualizations",
        "title": "Triangle Shapes",
        "description": "BFS expressions targeting triangle shapes of any color or style."
    },
    {
        "dir": "bfs_examples_circle_solid/visualizations",
        "title": "Solid Circles",
        "description": "BFS expressions targeting circles with solid fill style."
    },
    {
        "dir": "bfs_examples_half/visualizations",
        "title": "Big Objects with Half Style",
        "description": "BFS expressions targeting big objects with half-and-half fill style."
    }
]

def copy_images(src_dir: str, dest_dir: str, prefix: str = "") -> List[str]:
    """
    Copy BFS visualization images to the destination directory.

    Args:
        src_dir: Source directory containing visualization images
        dest_dir: Destination directory for copied images
        prefix: Prefix to add to copied image filenames

    Returns:
        List of paths to copied images (relative to dest_dir)
    """
    copied_images = []

    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Find all BFS visualization images in the source directory
    image_paths = glob.glob(os.path.join(src_dir, "BFS_*.png"))

    # Copy each image to the destination directory with a prefix
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        if "combined" in filename:
            dest_filename = f"{prefix}_combined.png"
        else:
            # Extract the index from the filename
            parts = filename.split("_")
            if len(parts) >= 3 and parts[-1].endswith(".png"):
                index = parts[-1].replace(".png", "")
                dest_filename = f"{prefix}_{index}.png"
            else:
                dest_filename = f"{prefix}_{i+1}.png"

        dest_path = os.path.join(dest_dir, dest_filename)
        shutil.copy2(path, dest_path)
        copied_images.append(dest_filename)

    return copied_images

def create_html_report(categories: List[Dict], output_dir: str, output_file: str = "bfs_visualizations_report.html") -> str:
    """
    Create an HTML report with all BFS visualizations.

    Args:
        categories: List of category dictionaries with info about visualizations
        output_dir: Directory to place the report and images
        output_file: Name of the HTML report file

    Returns:
        Path to the created HTML report
    """
    # Create images directory inside output_dir
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # HTML header
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>BFS Referring Expressions Visualization Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }}
        h1, h2 {{
            color: #333;
        }}
        h1 {{
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }}
        h2 {{
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .category {{
            margin-bottom: 30px;
        }}
        .description {{
            font-style: italic;
            color: #555;
            margin-bottom: 15px;
        }}
        .image-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }}
        .combined-visualization {{
            margin-top: 20px;
        }}
        .combined-visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }}
        footer {{
            margin-top: 50px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>BFS (Attribute-Based) Referring Expressions Visualization Report</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>This report shows visualizations of various BFS (attribute-based) referring expressions with different constraints.</p>
"""

    # Process each category
    for i, category in enumerate(categories):
        src_dir = category["dir"]
        if not os.path.exists(src_dir):
            continue

        # Copy images with a prefix based on the category
        prefix = f"cat{i+1}"
        copied_images = copy_images(src_dir, images_dir, prefix)

        # Skip if no images were copied
        if not copied_images:
            continue

        # Add category section to HTML
        html += f"""
    <div class="category">
        <h2>{category['title']}</h2>
        <div class="description">{category['description']}</div>
"""

        # Add individual images (excluding combined)
        individual_images = [img for img in copied_images if "combined" not in img]
        if individual_images:
            html += f"""
        <div class="image-container">
"""
            for img in individual_images:
                html += f"""            <img src="images/{img}" alt="{img}" />
"""
            html += f"""        </div>
"""

        # Add combined visualization if available
        combined_images = [img for img in copied_images if "combined" in img]
        if combined_images:
            html += f"""
        <div class="combined-visualization">
            <h3>Combined Visualization</h3>
            <img src="images/{combined_images[0]}" alt="Combined visualization" />
        </div>
"""

        html += f"""    </div>
"""

    # HTML footer
    html += f"""
    <footer>
        <p>Created by the Reason-Synth visualization tool.</p>
    </footer>
</body>
</html>
"""

    # Write the HTML to file
    report_path = os.path.join(output_dir, output_file)
    with open(report_path, 'w') as f:
        f.write(html)

    return report_path

def main():
    """Parse command line arguments and create the report."""
    parser = argparse.ArgumentParser(description="Compile BFS visualizations into an HTML report")
    parser.add_argument("--output_dir", type=str, default="bfs_report",
                        help="Directory to place the report and images")
    parser.add_argument("--output_file", type=str, default="bfs_visualizations_report.html",
                        help="Name of the HTML report file")
    args = parser.parse_args()

    # Create the report
    report_path = create_html_report(EXAMPLE_CATEGORIES, args.output_dir, args.output_file)
    print(f"Report created at: {report_path}")

if __name__ == "__main__":
    main()