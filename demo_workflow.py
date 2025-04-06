#!/usr/bin/env python
"""
Demo Workflow: Generate synthetic images, referring expressions, and visualize them.

This script demonstrates the full workflow from generating synthetic images to
creating referring expressions and visualizing them using the modularized code.

Usage:
    python demo_workflow.py --num_images 5 --output_dir demo_output
"""

import os
import argparse
import subprocess
import random
import json
from typing import List, Dict, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Demo workflow for referring expressions')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of synthetic images to generate')
    parser.add_argument('--output_dir', type=str, default='demo_output',
                        help='Output directory for the demo')
    parser.add_argument('--min_grid', type=int, default=2,
                        help='Minimum grid size (both rows and columns)')
    parser.add_argument('--max_grid', type=int, default=4,
                        help='Maximum grid size (both rows and columns)')
    parser.add_argument('--sampling_dfs_ratio', type=float, default=0.6,
                        help='Ratio of DFS expressions to total expressions')
    parser.add_argument('--sampling_existence_ratio', type=float, default=0.5,
                        help='Ratio of existence-based vs random sampling (0.0 = pure random, 1.0 = pure existence-based)')
    parser.add_argument('--bfs_pattern_type', type=str, default='patterns',
                        choices=['all', 'patterns'],
                        help='Pattern distribution type for BFS expressions')
    parser.add_argument('--bfs_ratio_single_attr', type=float, default=0.25,
                        help='Proportion of single attribute expressions')
    parser.add_argument('--bfs_ratio_two_attr', type=float, default=0.25,
                        help='Proportion of two attribute combinations')
    parser.add_argument('--bfs_ratio_three_attr', type=float, default=0.25,
                        help='Proportion of three attribute combinations')
    parser.add_argument('--bfs_ratio_four_attr', type=float, default=0.25,
                        help='Proportion of all attribute expressions')
    parser.add_argument('--num_vis_samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for more detailed processing information')
    return parser.parse_args()

def run_command(command):
    """Run a command with subprocess."""
    cmd_str = ' '.join(command)
    print(f"Running: {cmd_str}")
    subprocess.run(command, check=True)

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, 'images')
    annotation_dir = os.path.join(args.output_dir, 'annotations')
    visualization_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Step 1: Generate synthetic images
    print("\n=== Step 1: Generating Synthetic Images ===")
    run_command([
        'python', 'generate_samples.py',
        '--num-samples', str(args.num_images),
        '--output-dir', args.output_dir,
        '--min-grid', str(args.min_grid),
        '--max-grid', str(args.max_grid),
        '--seed', str(args.seed)
    ])

    # Step 2: Create referring expressions dataset
    print("\n=== Step 2: Creating Referring Expressions Dataset ===")
    cmd = [
        'python', 'create_referring_expressions_dataset.py',
        '--input', os.path.join(annotation_dir, 'dataset.jsonl'),
        '--output', os.path.join(annotation_dir, 'referring_expressions.jsonl'),
        '--sampling_dfs_ratio', str(args.sampling_dfs_ratio),
        '--sampling_existence_ratio', str(args.sampling_existence_ratio),
        '--bfs_pattern_type', args.bfs_pattern_type,
        '--bfs_ratio_single_attr', str(args.bfs_ratio_single_attr),
        '--bfs_ratio_two_attr', str(args.bfs_ratio_two_attr),
        '--bfs_ratio_three_attr', str(args.bfs_ratio_three_attr),
        '--bfs_ratio_four_attr', str(args.bfs_ratio_four_attr)
    ]

    # Add debug flag if enabled
    if args.debug:
        cmd.append('--debug')

    run_command(cmd)

    # Step 3: Visualize referring expressions
    print("\n=== Step 3: Visualizing Referring Expressions ===")
    run_command([
        'python', 'visualize_referring_expressions.py',
        '--dataset', os.path.join(annotation_dir, 'referring_expressions.jsonl'),
        '--image_dir', image_dir,
        '--save_dir', visualization_dir,
        '--num_samples', str(args.num_vis_samples)
    ])

    # Step 4: Analyze results
    print("\n=== Step 4: Analyzing Results ===")
    # Count referring expressions by type
    with open(os.path.join(annotation_dir, 'referring_expressions.jsonl'), 'r') as f:
        expressions = [json.loads(line) for line in f]

    dfs_count = sum(1 for expr in expressions if expr['expression_type'] == 'DFS')
    bfs_count = sum(1 for expr in expressions if expr['expression_type'] == 'BFS')

    # Count unique matches (1-to-1 vs 1-to-many)
    unique_matches = sum(1 for expr in expressions if len(expr['matching_objects']) == 1)
    multiple_matches = sum(1 for expr in expressions if len(expr['matching_objects']) > 1)

    print(f"Total referring expressions: {len(expressions)}")
    print(f"DFS expressions: {dfs_count} ({dfs_count/len(expressions):.1%})")
    print(f"BFS expressions: {bfs_count} ({bfs_count/len(expressions):.1%})")
    print(f"Unique matches (1-to-1): {unique_matches} ({unique_matches/len(expressions):.1%})")
    print(f"Multiple matches (1-to-many): {multiple_matches} ({multiple_matches/len(expressions):.1%})")

    print("\nWorkflow complete! Results are in:")
    print(f"  - Images: {image_dir}")
    print(f"  - Annotations: {annotation_dir}")
    print(f"  - Visualizations: {visualization_dir}")
    print(f"  - Combined visualization: {os.path.join(visualization_dir, 'visualization_combined.png')}")

if __name__ == "__main__":
    main()