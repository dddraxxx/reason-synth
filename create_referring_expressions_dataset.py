#!/usr/bin/env python
"""
Generate referring expressions dataset for the ReasonSynth large dataset

This script creates a JSONL dataset for referring expression tasks by:
1. Parsing the existing dataset.jsonl
2. Generating natural language referring expressions for objects
3. Verifying which objects match each expression
4. Creating a dataset with expressions that may match one or multiple objects

Key Features:
- Supports both positional (DFS) and attribute-based (BFS) referring expressions
- Maintains a configurable ratio of DFS:BFS expressions
- Handles three types of referring expressions:
  * Unique match: Expression matches exactly one object
  * Multiple match: Expression matches multiple objects (intentional ambiguity)
  * Empty match: Expression matches no objects (used for testing negative cases)
- Extracts requirements implicit in each referring expression
- Records bounding box coordinates for evaluation

Usage:
    python create_referring_expressions_dataset.py
        --input /path/to/dataset.jsonl
        --output /path/to/refer_exp_dataset.jsonl
        --sampling_dfs_ratio 0.7
        --sampling_existence_ratio 0.5
        --max_images 0  # 0 means use all images, otherwise limits to specified number

Requirements:
- The DFSExpressionHandler class from dfs_expression_handler.py
- The BFSExpressionHandler class from bfs_expression_handler.py
- All dependencies from the main repository

Future Development:
- Improve expression parsing logic for more accurate matching
- Add support for more complex referring expressions
- Implement better NLP techniques for extracting requirements
- Optimize performance for larger datasets
"""

import os
import json
import random
import argparse
import itertools
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from tqdm import tqdm, trange

# Import the expression handlers
from dfs_expression_handler import DFSExpressionHandler
from bfs_expression_handler import BFSExpressionHandler

# Configuration for grid sizes and expression counts
GRID_SIZE_TO_COUNT = {
    (2, 2): 2,
    (3, 3): 4,
    (4, 4): 8,
    (5, 5): 8,
    (6, 6): 12,
    (7, 7): 12,
    (8, 8): 12
}

# Add all missing grid size combinations
for r in range(2, 9):
    for c in range(2, 9):
        if (r, c) not in GRID_SIZE_TO_COUNT:
            # Interpolate based on area
            area = r * c
            if area <= 4:  # 2x2
                GRID_SIZE_TO_COUNT[(r, c)] = 1
            elif area <= 9:  # 3x3
                GRID_SIZE_TO_COUNT[(r, c)] = 2
            elif area <= 16:  # 4x4
                GRID_SIZE_TO_COUNT[(r, c)] = 4
            elif area <= 25:  # 5x5
                GRID_SIZE_TO_COUNT[(r, c)] = 8
            elif area <= 36:  # 6x6
                GRID_SIZE_TO_COUNT[(r, c)] = 12
            else:  # 7x7, 8x8, etc.
                GRID_SIZE_TO_COUNT[(r, c)] = 12

def process_dataset(dataset_path: str) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Process the dataset.jsonl file to extract images grouped by grid size.

    Args:
        dataset_path: Path to the dataset.jsonl file

    Returns:
        Dictionary mapping grid sizes to lists of image data
    """
    grid_size_to_images = defaultdict(list)
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            grid_size = tuple(data['scene']['grid_size'])
            # Add the objects list to the data
            data['objects'] = data['scene']['shapes']
            grid_size_to_images[grid_size].append(data)

    return grid_size_to_images

def get_expression_count(grid_size):
    """
    Get the number of expressions to generate for a given grid size.

    Args:
        grid_size: Tuple of (rows, columns)

    Returns:
        Number of expressions to generate for this grid size
    """
    # Limit to our defined grid sizes
    r = min(8, max(2, grid_size[0]))
    c = min(8, max(2, grid_size[1]))

    # Look up the count for this grid size
    return GRID_SIZE_TO_COUNT.get((r, c), 5)  # Default to 5 if not found

def debug_object_attributes(objects):
    """Debug function to print object attributes for troubleshooting."""
    print("\nDEBUG: Object Attributes")
    for i, obj in enumerate(objects):
        attrs = []
        if 'shape_type' in obj:
            attrs.append(f"shape={obj['shape_type']}")
        if 'size' in obj:
            attrs.append(f"size={obj['size']}")
        if 'color1' in obj:
            attrs.append(f"color1={obj['color1']}")
        if 'color2' in obj:
            attrs.append(f"color2={obj['color2']}")
        if 'style' in obj:
            attrs.append(f"style={obj['style']}")
        print(f"  Object {i+1}: {', '.join(attrs)}")
    print("")

def generate_referring_expressions_dataset(input_path: str, output_path: str,
                              sampling_dfs_ratio: float = 0.7,
                              sampling_existence_ratio: float = 0.5,
                              max_images: int = 0,
                              grid_position_min: Tuple[int, int] = (0, 0),
                              grid_position_max: Tuple[int, int] = (7, 7),
                              bfs_pattern_type: str = "patterns",
                              bfs_ratio_single_attr: float = 0.25,
                              bfs_ratio_two_attr: float = 0.25,
                              bfs_ratio_three_attr: float = 0.25,
                              bfs_ratio_four_attr: float = 0.25,
                              debug: bool = False):
    """
    Generate the referring expressions dataset.

    Args:
        input_path: Path to the input dataset.jsonl file
        output_path: Path to save the output referring expressions dataset
        sampling_dfs_ratio: Ratio of DFS expressions to total expressions (default: 0.7)
        sampling_existence_ratio: Ratio of existence-based vs random sampling (default: 0.5)
                                 0.0 = pure random, 1.0 = pure existence-based
        max_images: Maximum number of images to process (0 = use all images)
        grid_position_min: Minimum grid position (row, col) for DFS expressions
        grid_position_max: Maximum grid position (row, col) for DFS expressions
        bfs_pattern_type: Pattern distribution type for BFS expressions ("patterns" or "all")
        bfs_ratio_single_attr: Proportion of single attribute expressions (default: 0.25)
        bfs_ratio_two_attr: Proportion of two attribute combinations (default: 0.25)
        bfs_ratio_three_attr: Proportion of three attribute combinations (default: 0.25)
        bfs_ratio_four_attr: Proportion of all attribute expressions (default: 0.25)
        debug: Enable debug output
    """
    # Process input dataset
    grid_size_to_images = process_dataset(input_path)

    # Initialize the expression handlers
    dfs_handler = DFSExpressionHandler(max_grid_size=(8, 8))
    bfs_handler = BFSExpressionHandler()

    # Initialize counters
    total_expressions = 0
    dfs_count = 0
    bfs_count = 0

    # Existence vs random counts
    existence_expressions = 0
    random_expressions = 0

    # Limit images if requested
    total_images_processed = 0

    # Open the output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out_file:
        # Process images by grid size
        for grid_size, images in grid_size_to_images.items():
            # Limit number of images if requested
            if max_images > 0 and total_images_processed >= max_images:
                break

            # Get number of expressions to generate for this grid size
            num_expressions = get_expression_count(grid_size)

            # Process each image with this grid size
            for image_data in images:
                # Limit number of images if requested
                if max_images > 0 and total_images_processed >= max_images:
                    break

                image_path = image_data['image_path']
                objects = image_data['objects']

                # Skip if no objects in the image
                if not objects:
                    continue

                # Debug: print object attributes
                if debug:
                    print(f"\nProcessing image: {image_path}")
                    debug_object_attributes(objects)

                # Add grid position to objects if needed
                for obj in objects:
                    if 'grid_position' not in obj and 'row' in obj and 'col' in obj:
                        obj['grid_position'] = [obj['row'], obj['col']]

                # Calculate number of expressions of each type
                # Ensure we get at least one of each type if possible
                total_expr = max(2, num_expressions)  # Ensure we have at least 2 expressions
                num_dfs = max(1, int(total_expr * sampling_dfs_ratio))
                num_bfs = max(1, total_expr - num_dfs)

                if debug:
                    print(f"Target expressions: DFS={num_dfs}, BFS={num_bfs}, Total={total_expr}")

                # Store valid expressions
                valid_expressions = []

                # Generate DFS (position-based) expressions
                if num_dfs > 0:
                    # Ensure min/max grid values are within the current grid size
                    rows, cols = grid_size
                    image_min_grid = (max(0, min(grid_position_min[0], rows - 1)), max(0, min(grid_position_min[1], cols - 1)))
                    image_max_grid = (min(rows - 1, grid_position_max[0]), min(cols - 1, grid_position_max[1]))

                    dfs_expressions = dfs_handler.generate_dfs_referring_expressions(
                        num_dfs,
                        min_grid=image_min_grid,
                        max_grid=image_max_grid
                    )

                    # Find matching objects for each DFS expression
                    for expr_data in dfs_expressions:
                        matching_objects = dfs_handler.find_matching_objects(
                            expr_data['target_requirements'],
                            objects,
                            grid_size
                        )

                        # Add expression with its matching objects (even if empty)
                        expr_data['matching_objects'] = matching_objects
                        valid_expressions.append(expr_data)
                        dfs_count += 1

                        if debug:
                            if matching_objects:
                                print(f"  ✓ DFS expression '{expr_data['referring_expression']}' found {len(matching_objects)} matching objects")
                            else:
                                print(f"  ○ DFS expression '{expr_data['referring_expression']}' has no matches (empty match case)")

                    if debug:
                        print(f"Generated {len(dfs_expressions)} DFS expressions, all kept")

                # Generate BFS (attribute-based) expressions
                if num_bfs > 0:
                    # Generate many more BFS expressions than needed, to ensure we have enough valid ones
                    generation_multiplier = 3

                    # Calculate how many expressions to generate with existence vs random
                    existence_count = max(1, int(num_bfs * sampling_existence_ratio * generation_multiplier))
                    random_count = max(1, int(num_bfs * (1 - sampling_existence_ratio) * generation_multiplier))

                    all_bfs_expressions = []

                    # Handle existence-based expressions
                    if existence_count > 0:
                        # Use all objects for generating existence-based expressions
                        existence_bfs = bfs_handler.generate_bfs_expressions(
                            objects,
                            bfs_pattern_type="all" if bfs_pattern_type == "all" else "patterns",
                            samples_per_pattern=existence_count,
                            bfs_ratio_single_attr=bfs_ratio_single_attr,
                            bfs_ratio_two_attr=bfs_ratio_two_attr,
                            bfs_ratio_three_attr=bfs_ratio_three_attr,
                            bfs_ratio_four_attr=bfs_ratio_four_attr
                        )

                        if debug:
                            print(f"Generated {len(existence_bfs)} existence-based BFS expressions")

                        all_bfs_expressions.extend(existence_bfs)
                        existence_expressions += len(existence_bfs)

                    # Handle random-based expressions
                    if random_count > 0:
                        # Generate with random objects (will be filtered later)
                        random_bfs = bfs_handler.generate_bfs_expressions(
                            objects[:min(len(objects), 10)],  # Limit to 10 objects for efficiency
                            bfs_pattern_type="patterns",  # Always use patterns for random sampling
                            samples_per_pattern=random_count,
                            bfs_ratio_single_attr=bfs_ratio_single_attr,
                            bfs_ratio_two_attr=bfs_ratio_two_attr,
                            bfs_ratio_three_attr=bfs_ratio_three_attr,
                            bfs_ratio_four_attr=bfs_ratio_four_attr
                        )

                        if debug:
                            print(f"Generated {len(random_bfs)} random-based BFS expressions")

                        all_bfs_expressions.extend(random_bfs)
                        random_expressions += len(random_bfs)

                    # Find matching objects for each BFS expression
                    valid_bfs = []
                    for expr_data in all_bfs_expressions:
                        if debug and 'target_requirements' in expr_data:
                            print(f"Checking BFS expression: '{expr_data['referring_expression']}'")
                            print(f"  Requirements: {expr_data['target_requirements']}")

                        matching_objects = bfs_handler.find_matching_objects(
                            expr_data['target_requirements'],
                            objects
                        )

                        # Add expression with its matching objects (even if empty)
                        expr_data['matching_objects'] = matching_objects
                        valid_bfs.append(expr_data)
                        if debug:
                            if matching_objects:
                                print(f"  ✓ Found {len(matching_objects)} matching objects")
                            else:
                                print(f"  ○ No matching objects (empty match case)")

                    # Shuffle and limit to the required number
                    random.shuffle(valid_bfs)
                    valid_bfs = valid_bfs[:num_bfs]
                    valid_expressions.extend(valid_bfs)
                    bfs_count += len(valid_bfs)

                    if debug:
                        print(f"Generated {len(all_bfs_expressions)} BFS expressions, {len(valid_bfs)} valid")

                # Write expressions to output file
                for expr_data in valid_expressions:
                    # Create dataset entry
                    entry = {
                        'image_path': image_path,
                        'grid_size': list(grid_size),
                        'referring_expression': expr_data['referring_expression'],
                        'expression_type': expr_data['expression_type'],
                        'target_requirements': expr_data['target_requirements'],
                        'primary_target_idx': -1,  # No specific primary target
                        'matching_objects': expr_data['matching_objects']
                    }

                    # Write to output file
                    out_file.write(json.dumps(entry) + '\n')
                    total_expressions += 1

                # Increment image counter
                total_images_processed += 1

                # Print progress
                if total_images_processed % 10 == 0:
                    print(f"Processed {total_images_processed} images, generated {total_expressions} expressions")

    # Print summary
    print(f"\nDataset generation complete:")
    print(f"Total images processed: {total_images_processed}")
    print(f"Total expressions generated: {total_expressions}")
    print(f"DFS expressions: {dfs_count} ({dfs_count/max(1, total_expressions):.1%})")
    print(f"BFS expressions: {bfs_count} ({bfs_count/max(1, total_expressions):.1%})")
    if bfs_count > 0:
        print(f"  - Existence-based: {existence_expressions} ({existence_expressions/max(1, bfs_count):.1%} of BFS)")
        print(f"  - Random-based: {random_expressions} ({random_expressions/max(1, bfs_count):.1%} of BFS)")

    # Add match statistics - read from the output file
    with open(output_path, 'r') as f:
        all_expressions = [json.loads(line) for line in f]

    empty_matches = sum(1 for expr in all_expressions if not expr['matching_objects'])
    unique_matches = sum(1 for expr in all_expressions if len(expr['matching_objects']) == 1)
    multiple_matches = sum(1 for expr in all_expressions if len(expr['matching_objects']) > 1)

    print(f"\nMatching statistics:")
    print(f"  - Empty matches: {empty_matches} ({empty_matches/max(1, total_expressions):.1%})")
    print(f"  - Unique matches: {unique_matches} ({unique_matches/max(1, total_expressions):.1%})")
    print(f"  - Multiple matches: {multiple_matches} ({multiple_matches/max(1, total_expressions):.1%})")

    print(f"Output saved to: {output_path}")

def main():
    """Main function to parse arguments and generate the dataset."""
    parser = argparse.ArgumentParser(description="Generate referring expressions dataset")
    parser.add_argument("--input", required=True, help="Path to the input dataset.jsonl file")
    parser.add_argument("--output", required=True, help="Path to save the output referring expressions dataset")
    parser.add_argument("--sampling_dfs_ratio", type=float, default=0.7,
                        help="Ratio of DFS expressions to total expressions (default: 0.7)")
    parser.add_argument("--sampling_existence_ratio", type=float, default=0.5,
                        help="Ratio of existence to random sampling for BFS expressions (0.0=random, 1.0=existence)")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Maximum number of images to process (0 = use all)")

    # Add grid position parameters for DFS expressions
    parser.add_argument("--grid_position_min_row", type=int, default=0,
                        help="Minimum grid row for DFS expressions (0-indexed)")
    parser.add_argument("--grid_position_min_col", type=int, default=0,
                        help="Minimum grid column for DFS expressions (0-indexed)")
    parser.add_argument("--grid_position_max_row", type=int, default=7,
                        help="Maximum grid row for DFS expressions (0-indexed)")
    parser.add_argument("--grid_position_max_col", type=int, default=7,
                        help="Maximum grid column for DFS expressions (0-indexed)")

    # Add BFS pattern parameters
    parser.add_argument("--bfs_pattern_type", type=str, default="patterns", choices=["patterns", "all"],
                       help="Pattern distribution type for BFS expressions")
    parser.add_argument("--bfs_ratio_single_attr", type=float, default=0.25,
                       help="Proportion of single attribute expressions")
    parser.add_argument("--bfs_ratio_two_attr", type=float, default=0.25,
                       help="Proportion of two attribute combinations")
    parser.add_argument("--bfs_ratio_three_attr", type=float, default=0.25,
                       help="Proportion of three attribute combinations")
    parser.add_argument("--bfs_ratio_four_attr", type=float, default=0.25,
                       help="Proportion of all attribute expressions")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Set grid range parameters for DFS
    grid_position_min = (args.grid_position_min_row, args.grid_position_min_col)
    grid_position_max = (args.grid_position_max_row, args.grid_position_max_col)

    print(f"Generating referring expressions dataset...")
    print(f"Input dataset: {args.input}")
    print(f"Output dataset: {args.output}")
    print(f"Sampling DFS ratio: {args.sampling_dfs_ratio:.2f}")
    print(f"Sampling existence ratio: {args.sampling_existence_ratio:.2f}")
    print(f"Maximum images to process: {args.max_images if args.max_images > 0 else 'all'}")
    print(f"Grid position range: {grid_position_min} to {grid_position_max}")
    print(f"BFS pattern type: {args.bfs_pattern_type}")
    print(f"BFS attribute ratios: single={args.bfs_ratio_single_attr:.2f}, two={args.bfs_ratio_two_attr:.2f}, "
          f"three={args.bfs_ratio_three_attr:.2f}, four={args.bfs_ratio_four_attr:.2f}")
    if args.debug:
        print("Debug output enabled")

    generate_referring_expressions_dataset(
        args.input,
        args.output,
        sampling_dfs_ratio=args.sampling_dfs_ratio,
        sampling_existence_ratio=args.sampling_existence_ratio,
        max_images=args.max_images,
        grid_position_min=grid_position_min,
        grid_position_max=grid_position_max,
        bfs_pattern_type=args.bfs_pattern_type,
        bfs_ratio_single_attr=args.bfs_ratio_single_attr,
        bfs_ratio_two_attr=args.bfs_ratio_two_attr,
        bfs_ratio_three_attr=args.bfs_ratio_three_attr,
        bfs_ratio_four_attr=args.bfs_ratio_four_attr,
        debug=args.debug
    )

if __name__ == "__main__":
    main()