#!/usr/bin/env python
"""
Generate referring expressions dataset with controlled reasoning levels for the ReasonSynth dataset

This script creates a JSONL dataset for referring expression tasks by:
1. Parsing the existing dataset.jsonl
2. Generating natural language referring expressions with specific reasoning levels
3. Verifying which objects match each expression
4. Creating a dataset with controlled reasoning level distribution

Key Features:
- Supports both positional (DFS) and attribute-based (BFS) referring expressions
- Maintains a configurable ratio of DFS:BFS expressions
- Allows specifying the distribution of reasoning levels in generated expressions
- Limits reasoning levels to reasonable maximums (12 for DFS, 5 for BFS)
- Records reasoning level for each expression for analysis
- Handles three types of referring expressions:
  * Unique match: Expression matches exactly one object
  * Multiple match: Expression matches multiple objects (intentional ambiguity)
  * Empty match: Expression matches no objects (used for testing negative cases)

Usage:
    python create_referring_expressions_with_reasoning_levels.py
        --input /path/to/dataset.jsonl
        --output /path/to/refer_exp_dataset.jsonl
        --sampling_dfs_ratio 0.7
        --sampling_existence_ratio 0.5
        --dfs_reasoning_levels 2:0.3,4:0.3,6:0.4
        --bfs_reasoning_levels 1:0.2,2:0.3,3:0.3,4:0.2
        --max_images 0  # 0 means use all images

Requirements:
- The DFSExpressionHandler class from dfs_expression_handler.py
- The BFSExpressionHandler class from bfs_expression_handler.py
"""

import os
import json
import random
import argparse
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from tqdm import tqdm

# Import the expression handlers
from dfs_expression_handler import DFSExpressionHandler
from bfs_expression_handler import BFSExpressionHandler, SHAPE_TYPES, COLORS, SIZES, STYLES

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

# Maximum reasoning levels
MAX_DFS_REASONING_LEVEL = 12
MAX_BFS_REASONING_LEVEL = 5

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

def parse_reasoning_level_ratios(ratio_str: Optional[str], max_level: int) -> Dict[int, float]:
    """
    Parse the reasoning level ratio string into a dictionary.

    Format: "level1:ratio1,level2:ratio2,..."
    Example: "1:0.2,2:0.3,3:0.5"

    Args:
        ratio_str: String with reasoning level ratios or None
        max_level: Maximum reasoning level to consider

    Returns:
        Dictionary mapping reasoning levels to their ratios
    """
    if not ratio_str:
        # If no ratios provided, distribute evenly among reasonable levels
        # For DFS: levels 2-8 (common grid positions)
        # For BFS: levels 1-4 (common attribute counts)
        if max_level == MAX_DFS_REASONING_LEVEL:
            # DFS typically starts at level 2 (1 row + 1 column)
            levels = list(range(2, min(9, max_level + 1)))
        else:  # BFS
            # BFS starts at level 1 (single attribute)
            levels = list(range(1, min(5, max_level + 1)))

        ratio = 1.0 / len(levels)
        return {level: ratio for level in levels}

    result = {}
    total_ratio = 0.0

    # Parse the ratio string
    parts = ratio_str.split(',')
    for part in parts:
        try:
            level_str, ratio_str = part.split(':')
            level = int(level_str.strip())
            ratio = float(ratio_str.strip())

            if level < 0 or level > max_level:
                print(f"Warning: Reasoning level {level} outside valid range (1-{max_level}). Ignoring.")
                continue

            if ratio < 0:
                print(f"Warning: Negative ratio {ratio} for level {level}. Setting to 0.")
                ratio = 0

            result[level] = ratio
            total_ratio += ratio
        except ValueError:
            print(f"Warning: Could not parse reasoning level ratio '{part}'. Expected format 'level:ratio'.")

    # Normalize ratios if they don't sum to 1
    if result and abs(total_ratio - 1.0) > 1e-6:
        if total_ratio == 0:
            print("Warning: All ratios are zero. Distributing evenly.")
            ratio = 1.0 / len(result)
            for level in result:
                result[level] = ratio
        else:
            for level in result:
                result[level] /= total_ratio

    return result

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

def allocate_expression_counts(total: int, level_ratios: Dict[int, float]) -> Dict[int, int]:
    """
    Allocate a total number of expressions according to level ratios.

    Args:
        total: Total number of expressions to allocate
        level_ratios: Dictionary mapping reasoning levels to their ratios

    Returns:
        Dictionary mapping reasoning levels to expression counts
    """
    result = {}
    remaining = total

    # First pass: allocate based on ratios, storing fractional parts
    fractional_loss = 0
    for level, ratio in level_ratios.items():
        count = total * ratio
        int_count = int(count)
        fractional_loss += count - int_count
        result[level] = int_count
        remaining -= int_count

    # Distribute remaining expressions to levels with highest fractional loss
    if remaining > 0 and level_ratios:
        # Create a list of (level, fractional_part) pairs
        fractions = [(level, total * ratio - result[level]) for level, ratio in level_ratios.items()]
        # Sort by fractional part in descending order
        fractions.sort(key=lambda x: x[1], reverse=True)

        # Distribute remaining counts
        for i in range(min(remaining, len(fractions))):
            result[fractions[i][0]] += 1

    return result

def generate_referring_expressions_dataset(
    input_path: str,
    output_path: str,
    sampling_dfs_ratio: float = 0.7,
    sampling_existence_ratio: float = 0.5,
    dfs_reasoning_levels: Optional[str] = None,
    bfs_reasoning_levels: Optional[str] = None,
    max_images: int = 0,
    grid_position_min: Tuple[int, int] = (0, 0),
    grid_position_max: Tuple[int, int] = (7, 7),
    debug: bool = False
):
    """
    Generate the referring expressions dataset with controlled reasoning levels.

    Args:
        input_path: Path to the input dataset.jsonl file
        output_path: Path to save the output referring expressions dataset
        sampling_dfs_ratio: Ratio of DFS expressions to total expressions (default: 0.7)
        sampling_existence_ratio: Ratio of existence-based vs random sampling (default: 0.5)
        dfs_reasoning_levels: String specifying DFS reasoning level ratios (e.g., "2:0.3,4:0.3,6:0.4")
        bfs_reasoning_levels: String specifying BFS reasoning level ratios (e.g., "1:0.2,2:0.3,3:0.3,4:0.2")
        max_images: Maximum number of images to process (0 = use all images)
        grid_position_min: Minimum grid position (row, col) for DFS expressions
        grid_position_max: Maximum grid position (row, col) for DFS expressions
        debug: Enable debug output
    """
    # Process input dataset
    grid_size_to_images = process_dataset(input_path)

    # Parse reasoning level ratios
    dfs_level_ratios = parse_reasoning_level_ratios(dfs_reasoning_levels, MAX_DFS_REASONING_LEVEL)
    bfs_level_ratios = parse_reasoning_level_ratios(bfs_reasoning_levels, MAX_BFS_REASONING_LEVEL)

    # Print reasoning level configurations
    print("\nReasoning Level Distribution:")
    print("DFS Reasoning Levels:")
    for level, ratio in sorted(dfs_level_ratios.items()):
        print(f"  Level {level}: {ratio:.1%}")

    print("BFS Reasoning Levels:")
    for level, ratio in sorted(bfs_level_ratios.items()):
        print(f"  Level {level}: {ratio:.1%}")

    # Initialize the expression handlers
    dfs_handler = DFSExpressionHandler(max_grid_size=(8, 8))
    bfs_handler = BFSExpressionHandler()

    # Initialize counters
    total_expressions = 0
    dfs_count = 0
    bfs_count = 0

    # Reasoning level statistics
    dfs_level_counts = defaultdict(int)
    bfs_level_counts = defaultdict(int)

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
                total_expr = max(1, num_expressions)  # Ensure we have at least 2 expressions
                num_dfs = max(0, int(total_expr * sampling_dfs_ratio))
                num_bfs = max(0, total_expr - num_dfs)

                if debug:
                    print(f"Target expressions: DFS={num_dfs}, BFS={num_bfs}, Total={total_expr}")

                # Distribute DFS expressions by reasoning level
                dfs_counts_by_level = allocate_expression_counts(num_dfs, dfs_level_ratios)

                # Distribute BFS expressions by reasoning level
                bfs_counts_by_level = allocate_expression_counts(num_bfs, bfs_level_ratios)

                if debug:
                    print("Expression counts by reasoning level:")
                    print("DFS:", dfs_counts_by_level)
                    print("BFS:", bfs_counts_by_level)

                # Store valid expressions
                valid_expressions = []

                # Generate DFS (position-based) expressions for each reasoning level
                if num_dfs > 0:
                    # Ensure min/max grid values are within the current grid size
                    rows, cols = grid_size
                    image_min_grid = (max(0, min(grid_position_min[0], rows - 1)),
                                      max(0, min(grid_position_min[1], cols - 1)))
                    image_max_grid = (min(rows - 1, grid_position_max[0]),
                                      min(cols - 1, grid_position_max[1]))

                    for reasoning_level, count in dfs_counts_by_level.items():
                        if count <= 0:
                            continue

                        # Split between existence-based and random expressions
                        existence_count = max(1, int(count * sampling_existence_ratio))
                        random_count = max(0, count - existence_count)

                        all_level_expressions = []

                        # Generate random position expressions for this reasoning level
                        if random_count > 0:
                            dfs_expressions = dfs_handler.generate_dfs_referring_expressions(
                                random_count,
                                min_grid=image_min_grid,
                                max_grid=image_max_grid,
                                reasoning_level=reasoning_level
                            )

                            # Add source field
                            for expr in dfs_expressions:
                                expr['source'] = 'random'

                            all_level_expressions.extend(dfs_expressions)

                        # Generate existence-based expressions for this reasoning level
                        if existence_count > 0 and objects:
                            # Extract grid positions from existing objects
                            existing_positions = []
                            for obj in objects:
                                if 'grid_position' in obj:
                                    pos = tuple(obj['grid_position'])
                                    if image_min_grid[0] <= pos[0] <= image_max_grid[0] and \
                                       image_min_grid[1] <= pos[1] <= image_max_grid[1]:
                                        # Calculate if this position would yield the target reasoning level
                                        display_row = pos[0] + 1
                                        display_col = pos[1] + 1
                                        if (display_row + display_col) - 2 == reasoning_level:
                                            existing_positions.append(pos)

                            # Generate expressions from valid existing positions
                            existence_expressions_generated = []
                            if existing_positions:
                                # Try to generate up to existence_count expressions
                                for _ in range(min(existence_count, len(existing_positions))):
                                    # Choose a random position
                                    if not existing_positions:
                                        break

                                    pos_idx = random.randrange(len(existing_positions))
                                    row, col = existing_positions.pop(pos_idx)

                                    # Generate expression for this position
                                    expr = dfs_handler.generate_dfs_referring_expressions(
                                        1,
                                        min_grid=(row, col),
                                        max_grid=(row, col)
                                    )

                                    # Verify the reasoning level matches
                                    if expr and expr[0]['reasoning_level'] == reasoning_level:
                                        # Add source field
                                        expr[0]['source'] = 'existence'
                                        existence_expressions_generated.extend(expr)

                            # If we couldn't generate enough existence-based expressions,
                            # supplement with more random ones
                            if len(existence_expressions_generated) < existence_count:
                                additional_needed = existence_count - len(existence_expressions_generated)
                                if additional_needed > 0:
                                    additional_exprs = dfs_handler.generate_dfs_referring_expressions(
                                        additional_needed,
                                        min_grid=image_min_grid,
                                        max_grid=image_max_grid,
                                        reasoning_level=reasoning_level
                                    )

                                    # Mark these as random even though they're filling existence quota
                                    for expr in additional_exprs:
                                        expr['source'] = 'random'

                                    existence_expressions_generated.extend(additional_exprs)

                            all_level_expressions.extend(existence_expressions_generated)

                        # Find matching objects for each expression at this reasoning level
                        for expr_data in all_level_expressions:
                            matching_objects = dfs_handler.find_matching_objects(
                                expr_data['target_requirements'],
                                objects,
                                grid_size
                            )

                            # Add expression with its matching objects
                            expr_data['matching_objects'] = matching_objects
                            valid_expressions.append(expr_data)
                            dfs_count += 1
                            dfs_level_counts[reasoning_level] += 1

                            if debug:
                                if matching_objects:
                                    print(f"  ✓ DFS L{reasoning_level} '{expr_data['referring_expression']}' found {len(matching_objects)} matches")
                                else:
                                    print(f"  ○ DFS L{reasoning_level} '{expr_data['referring_expression']}' has no matches")

                # Generate BFS (attribute-based) expressions for each reasoning level
                if num_bfs > 0:
                    for reasoning_level, count in bfs_counts_by_level.items():
                        if count <= 0:
                            continue

                        # Split between existence-based and random expressions
                        existence_count = max(1, int(count * sampling_existence_ratio))
                        random_count = max(0, count - existence_count)

                        all_level_expressions = []

                        # Generate existence-based expressions for this reasoning level
                        if existence_count > 0:
                            existence_bfs = bfs_handler.generate_bfs_expressions(
                                objects,
                                bfs_pattern_type="all",  # Use all to maximize chance of finding matching expressions
                                num_expressions=existence_count,
                                reasoning_level=reasoning_level
                            )

                            # Add source field
                            for expr in existence_bfs:
                                expr['source'] = 'existence'

                            all_level_expressions.extend(existence_bfs)
                            existence_expressions += len(existence_bfs)

                            if debug:
                                print(f"Generated {len(existence_bfs)} existence-based BFS level {reasoning_level} expressions")

                        # Generate random-based expressions for this reasoning level
                        if random_count > 0:
                            random_objects = bfs_handler.generate_random_objects(count=random_count * 2)  # Generate more to ensure enough variety

                            random_bfs = bfs_handler.generate_bfs_expressions(
                                random_objects,
                                num_expressions=random_count,
                                reasoning_level=reasoning_level
                            )

                            # Add source field
                            for expr in random_bfs:
                                expr['source'] = 'random'

                            all_level_expressions.extend(random_bfs)
                            random_expressions += len(random_bfs)

                            if debug:
                                print(f"Generated {len(random_bfs)} random-based BFS level {reasoning_level} expressions")

                        # Find matching objects for each expression at this reasoning level
                        for expr_data in all_level_expressions:
                            matching_objects = bfs_handler.find_matching_objects(
                                expr_data['target_requirements'],
                                objects
                            )

                            # Add expression with its matching objects
                            expr_data['matching_objects'] = matching_objects
                            valid_expressions.append(expr_data)
                            bfs_count += 1
                            bfs_level_counts[reasoning_level] += 1

                            if debug:
                                if matching_objects:
                                    print(f"  ✓ BFS L{reasoning_level} '{expr_data['referring_expression']}' found {len(matching_objects)} matches")
                                else:
                                    print(f"  ○ BFS L{reasoning_level} '{expr_data['referring_expression']}' has no matches")

                # Write expressions to output file
                for expr_data in valid_expressions:
                    # Create dataset entry
                    entry = {
                        'image_path': image_path,
                        'grid_size': list(grid_size),
                        'referring_expression': expr_data['referring_expression'],
                        'expression_type': expr_data['expression_type'],
                        'reasoning_level': expr_data['reasoning_level'],
                        'target_requirements': expr_data['target_requirements'],
                        'primary_target_idx': -1,  # No specific primary target
                        'matching_objects': expr_data['matching_objects']
                    }

                    # Preserve source field if it exists
                    if 'source' in expr_data:
                        entry['source'] = expr_data['source']

                    # Write to output file
                    out_file.write(json.dumps(entry) + '\n')
                    total_expressions += 1

                # Increment image counter
                total_images_processed += 1

                # Print progress
                # if total_images_processed % 10 == 0:
                #     print(f"Processed {total_images_processed} images, generated {total_expressions} expressions")

    # Print summary
    print(f"\nDataset generation complete:")
    print(f"Total images processed: {total_images_processed}")
    print(f"Total expressions generated: {total_expressions}")
    print(f"DFS expressions: {dfs_count} ({dfs_count/max(1, total_expressions):.1%})")
    print(f"BFS expressions: {bfs_count} ({bfs_count/max(1, total_expressions):.1%})")

    # Print reasoning level distribution
    print("\nReasoning level distribution:")
    print("DFS:")
    for level in sorted(dfs_level_counts.keys()):
        count = dfs_level_counts[level]
        print(f"  Level {level}: {count} ({count/max(1, dfs_count):.1%})")

    print("BFS:")
    for level in sorted(bfs_level_counts.keys()):
        count = bfs_level_counts[level]
        print(f"  Level {level}: {count} ({count/max(1, bfs_count):.1%})")

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
    parser = argparse.ArgumentParser(description="Generate referring expressions dataset with controlled reasoning levels")
    parser.add_argument("--input", required=True, help="Path to the input dataset.jsonl file")
    parser.add_argument("--output", required=True, help="Path to save the output referring expressions dataset")
    parser.add_argument("--sampling_dfs_ratio", type=float, default=0.7,
                        help="Ratio of DFS expressions to total expressions (default: 0.7)")
    parser.add_argument("--sampling_existence_ratio", type=float, default=0.5,
                        help="Ratio of existence to random sampling (0.0=random, 1.0=existence)")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Maximum number of images to process (0 = use all)")

    # Add reasoning level parameters
    parser.add_argument("--dfs_reasoning_levels", type=str, default=None,
                        help="Comma-separated list of DFS reasoning level:ratio pairs (e.g., '2:0.3,4:0.3,6:0.4')")
    parser.add_argument("--bfs_reasoning_levels", type=str, default=None,
                        help="Comma-separated list of BFS reasoning level:ratio pairs (e.g., '1:0.2,2:0.3,3:0.3,4:0.2')")

    # Add grid position parameters for DFS expressions
    parser.add_argument("--refexp_grid_min_row", type=int, default=0,
                        help="Minimum grid row for DFS referring expressions (0-indexed)")
    parser.add_argument("--refexp_grid_min_col", type=int, default=0,
                        help="Minimum grid column for DFS referring expressions (0-indexed)")
    parser.add_argument("--refexp_grid_max_row", type=int, default=7,
                        help="Maximum grid row for DFS referring expressions (0-indexed)")
    parser.add_argument("--refexp_grid_max_col", type=int, default=7,
                        help="Maximum grid column for DFS referring expressions (0-indexed)")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Set grid range parameters for DFS
    grid_position_min = (args.refexp_grid_min_row, args.refexp_grid_min_col)
    grid_position_max = (args.refexp_grid_max_row, args.refexp_grid_max_col)

    print(f"Generating referring expressions dataset with controlled reasoning levels...")
    print(f"Input dataset: {args.input}")
    print(f"Output dataset: {args.output}")
    print(f"Sampling DFS ratio: {args.sampling_dfs_ratio:.2f}")
    print(f"Sampling existence ratio: {args.sampling_existence_ratio:.2f}")
    print(f"Maximum images to process: {args.max_images if args.max_images > 0 else 'all'}")
    print(f"Grid position range: {grid_position_min} to {grid_position_max}")
    print(f"DFS reasoning levels config: {args.dfs_reasoning_levels if args.dfs_reasoning_levels else 'default (evenly distributed)'}")
    print(f"BFS reasoning levels config: {args.bfs_reasoning_levels if args.bfs_reasoning_levels else 'default (evenly distributed)'}")
    if args.debug:
        print("Debug output enabled")

    generate_referring_expressions_dataset(
        args.input,
        args.output,
        sampling_dfs_ratio=args.sampling_dfs_ratio,
        sampling_existence_ratio=args.sampling_existence_ratio,
        dfs_reasoning_levels=args.dfs_reasoning_levels,
        bfs_reasoning_levels=args.bfs_reasoning_levels,
        max_images=args.max_images,
        grid_position_min=grid_position_min,
        grid_position_max=grid_position_max,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
