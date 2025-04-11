#!/usr/bin/env python
"""
Demo Script: Generate referring expressions with controlled reasoning levels

This script demonstrates how to use the create_referring_expressions_with_reasoning_levels.py
to generate datasets with specific distributions of reasoning levels.

Usage:
    python demo_reasoning_levels.py --case basic
    python demo_reasoning_levels.py --case dfs_progression
    python demo_reasoning_levels.py --case bfs_progression
    python demo_reasoning_levels.py --case mixed_levels
"""

import os
import argparse
import subprocess
import random
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Define the test cases with different reasoning level configurations
TEST_CASES = {
    "basic": {
        "description": "Basic case with equal distribution of reasoning levels",
        "dfs_reasoning_levels": None,  # Use default even distribution
        "bfs_reasoning_levels": None,  # Use default even distribution
        "sampling_dfs_ratio": 0.5,
        "sampling_existence_ratio": 0.5
    },
    "dfs_progression": {
        "description": "DFS expressions with progressively increasing difficulty",
        "dfs_reasoning_levels": "2:0.2,4:0.3,6:0.3,8:0.15,10:0.05", # Favor medium levels
        "bfs_reasoning_levels": None,
        "sampling_dfs_ratio": 0.7,
        "sampling_existence_ratio": 0.6
    },
    "bfs_progression": {
        "description": "BFS expressions with focus on specific reasoning levels",
        "dfs_reasoning_levels": None,
        "bfs_reasoning_levels": "1:0.3,2:0.4,3:0.2,4:0.1", # Focus on simpler expressions
        "sampling_dfs_ratio": 0.3,
        "sampling_existence_ratio": 0.6
    },
    "mixed_levels": {
        "description": "Mixed distribution with specific reasoning levels for both DFS and BFS",
        "dfs_reasoning_levels": "2:0.2,4:0.6,6:0.2", # Focus on medium difficulty
        "bfs_reasoning_levels": "1:0.3,3:0.7", # Focus on single and triple attribute expressions
        "sampling_dfs_ratio": 0.5,
        "sampling_existence_ratio": 0.6
    },
    "extreme_simple": {
        "description": "Extremely simple expressions only",
        "dfs_reasoning_levels": "0:0.3,1:0.3,2:0.4",
        "bfs_reasoning_levels": "1:1.0", # Only single attribute expressions
        "sampling_dfs_ratio": 0.6,
        "sampling_existence_ratio": 0.5
    },
    "extreme_simple_dfs": {
        "description": "Extremely simple DFS expressions only",
        "dfs_reasoning_levels": "0:0.3,1:0.3,2:0.4",
        "bfs_reasoning_levels": None,
        "sampling_dfs_ratio": 1,
        "sampling_existence_ratio": 0.5
    },
    "extreme_complex": {
        "description": "Complex expressions only",
        "dfs_reasoning_levels": "8:0.5,10:0.5", # Only difficult position expressions
        "bfs_reasoning_levels": "3:0.5,4:0.5", # Only multi-attribute expressions
        "sampling_dfs_ratio": 0.5,
        "sampling_existence_ratio": 0.5
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for referring expressions with reasoning levels')

    # Case selection
    parser.add_argument('--case', type=str, default='basic',
                        choices=list(TEST_CASES.keys()),
                        help='Test case to run')

    # General parameters
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of synthetic images to generate')
    parser.add_argument('--output_dir', type=str, default='demo_reasoning_levels',
                        help='Output directory for the demo')
    parser.add_argument('--min_grid', type=int, default=3,
                        help='Minimum grid size (both rows and columns)')
    parser.add_argument('--max_grid', type=int, default=6,
                        help='Maximum grid size (both rows and columns)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Visualization parameters
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Number of samples to visualize')

    # Debug parameter
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for more detailed processing information')

    return parser.parse_args()

def run_command(command):
    """Run a command with subprocess."""
    cmd_str = ' '.join(command)
    print(f"Running: {cmd_str}")
    subprocess.run(command, check=True)

def visualize_reasoning_level_distribution(expressions, output_dir, case_name):
    """
    Create visualizations showing the distribution of reasoning levels in the dataset.

    Args:
        expressions: List of referring expressions
        output_dir: Output directory for saving visualizations
        case_name: Name of the test case
    """
    # Count expressions by type and reasoning level
    dfs_levels = defaultdict(int)
    bfs_levels = defaultdict(int)

    for expr in expressions:
        expr_type = expr['expression_type']
        level = expr['reasoning_level']

        if expr_type == 'DFS':
            dfs_levels[level] += 1
        elif expr_type == 'BFS':
            bfs_levels[level] += 1

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot DFS distributions
    if dfs_levels:
        levels = sorted(dfs_levels.keys())
        counts = [dfs_levels[lvl] for lvl in levels]
        bars = ax1.bar(levels, counts, color='skyblue')
        ax1.set_xlabel('Reasoning Level')
        ax1.set_ylabel('Count')
        ax1.set_title('DFS Expression Reasoning Levels')
        ax1.set_xticks(levels)
        for i, count in enumerate(counts):
            ax1.text(levels[i], count + 0.5, str(count), ha='center')

        # Remove the box/spines around the plot
        for spine in ax1.spines.values():
            spine.set_visible(False)
    else:
        ax1.text(0.5, 0.5, 'No DFS expressions', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('DFS Expression Reasoning Levels (None)')
        ax1.axis('off')

    # Plot BFS distributions
    if bfs_levels:
        levels = sorted(bfs_levels.keys())
        counts = [bfs_levels[lvl] for lvl in levels]
        bars = ax2.bar(levels, counts, color='salmon')
        ax2.set_xlabel('Reasoning Level')
        ax2.set_ylabel('Count')
        ax2.set_title('BFS Expression Reasoning Levels')
        ax2.set_xticks(levels)
        for i, count in enumerate(counts):
            ax2.text(levels[i], count + 0.5, str(count), ha='center')

        # Remove the box/spines around the plot
        for spine in ax2.spines.values():
            spine.set_visible(False)
    else:
        ax2.text(0.5, 0.5, 'No BFS expressions', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('BFS Expression Reasoning Levels (None)')
        ax2.axis('off')

    # Set overall title
    fig.suptitle(f'Reasoning Level Distribution - Case: {case_name}', fontsize=16)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, f'reasoning_levels_{case_name}.png'))
    plt.close()

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Get the test case configuration
    case_config = TEST_CASES[args.case]

    print(f"\n=== Running Test Case: {args.case} ===")
    print(f"Description: {case_config['description']}")
    print(f"DFS Reasoning Levels: {case_config['dfs_reasoning_levels'] or 'Default (even distribution)'}")
    print(f"BFS Reasoning Levels: {case_config['bfs_reasoning_levels'] or 'Default (even distribution)'}")

    # Create output directories
    case_output_dir = os.path.join(args.output_dir, args.case)
    os.makedirs(case_output_dir, exist_ok=True)
    image_dir = os.path.join(case_output_dir, 'images')
    annotation_dir = os.path.join(case_output_dir, 'annotations')
    visualization_dir = os.path.join(case_output_dir, 'visualizations')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Step 1: Generate synthetic images
    print("\n=== Step 1: Generating Synthetic Images ===")
    run_command([
        'python', 'generate_samples.py',
        '--num-samples', str(args.num_images),
        '--output-dir', case_output_dir,
        '--min-grid', str(args.min_grid),
        '--max-grid', str(args.max_grid),
        '--seed', str(args.seed)
    ])

    # Step 2: Create referring expressions dataset with reasoning levels
    print("\n=== Step 2: Creating Referring Expressions Dataset with Reasoning Levels ===")
    cmd = [
        'python', 'create_referring_expressions_with_reasoning_levels.py',
        '--input', os.path.join(annotation_dir, 'dataset.jsonl'),
        '--output', os.path.join(annotation_dir, 'referring_expressions.jsonl'),
        '--sampling_dfs_ratio', str(case_config['sampling_dfs_ratio']),
        '--sampling_existence_ratio', str(case_config['sampling_existence_ratio']),
    ]

    # Add reasoning level parameters if provided
    if case_config['dfs_reasoning_levels']:
        cmd.extend(['--dfs_reasoning_levels', case_config['dfs_reasoning_levels']])

    if case_config['bfs_reasoning_levels']:
        cmd.extend(['--bfs_reasoning_levels', case_config['bfs_reasoning_levels']])

    # Add debug flag if enabled
    if args.debug:
        cmd.append('--debug')

    run_command(cmd)

    # Step 3: Visualize referring expressions
    print("\n=== Step 3: Visualizing Referring Expressions ===")
    try:
        run_command([
            'python', 'visualize_referring_expressions.py',
            '--dataset', os.path.join(annotation_dir, 'referring_expressions.jsonl'),
            '--image_dir', image_dir,
            '--save_dir', visualization_dir,
            '--num_samples', str(args.num_vis_samples)
        ])
    except subprocess.CalledProcessError:
        print("Warning: Visualization failed. This might be due to the absence of the visualization script.")

    # Step 4: Analyze results
    print("\n=== Step 4: Analyzing Results ===")
    referring_expressions_path = os.path.join(annotation_dir, 'referring_expressions.jsonl')
    if os.path.exists(referring_expressions_path):
        with open(referring_expressions_path, 'r') as f:
            expressions = [json.loads(line) for line in f]

        # General statistics
        dfs_count = sum(1 for expr in expressions if expr['expression_type'] == 'DFS')
        bfs_count = sum(1 for expr in expressions if expr['expression_type'] == 'BFS')
        total_count = len(expressions)

        # Match type statistics
        empty_matches = sum(1 for expr in expressions if len(expr['matching_objects']) == 0)
        unique_matches = sum(1 for expr in expressions if len(expr['matching_objects']) == 1)
        multiple_matches = sum(1 for expr in expressions if len(expr['matching_objects']) > 1)

        # Source statistics
        existence_bfs = sum(1 for expr in expressions if expr.get('expression_type') == 'BFS' and expr.get('source') == 'existence')
        random_bfs = sum(1 for expr in expressions if expr.get('expression_type') == 'BFS' and expr.get('source') == 'random')
        existence_dfs = sum(1 for expr in expressions if expr.get('expression_type') == 'DFS' and expr.get('source') == 'existence')
        random_dfs = sum(1 for expr in expressions if expr.get('expression_type') == 'DFS' and expr.get('source') == 'random')

        # Display statistics
        print(f"Total referring expressions: {total_count}")
        print(f"DFS expressions: {dfs_count} ({dfs_count/total_count:.1%} of total)")
        if dfs_count > 0:
            print(f"  - Existence-based: {existence_dfs} ({existence_dfs/dfs_count:.1%} of DFS)")
            print(f"  - Random-based: {random_dfs} ({random_dfs/dfs_count:.1%} of DFS)")

        print(f"BFS expressions: {bfs_count} ({bfs_count/total_count:.1%} of total)")
        if bfs_count > 0:
            print(f"  - Existence-based: {existence_bfs} ({existence_bfs/bfs_count:.1%} of BFS)")
            print(f"  - Random-based: {random_bfs} ({random_bfs/bfs_count:.1%} of BFS)")

        print("\nMatching statistics:")
        print(f"  - Empty matches: {empty_matches} ({empty_matches/total_count:.1%})")
        print(f"  - Unique matches: {unique_matches} ({unique_matches/total_count:.1%})")
        print(f"  - Multiple matches: {multiple_matches} ({multiple_matches/total_count:.1%})")

        # Reasoning level statistics and visualization
        print("\nReasoning level statistics:")

        # Group by expression type and reasoning level
        dfs_levels = defaultdict(int)
        bfs_levels = defaultdict(int)

        for expr in expressions:
            expr_type = expr['expression_type']
            level = expr['reasoning_level']

            if expr_type == 'DFS':
                dfs_levels[level] += 1
            elif expr_type == 'BFS':
                bfs_levels[level] += 1

        for level in sorted(dfs_levels.keys()):
            print(f"  - DFS Level {level}: {dfs_levels[level]}")
        for level in sorted(bfs_levels.keys()):
            print(f"  - BFS Level {level}: {bfs_levels[level]}")

        # Visualize the distribution
        visualize_reasoning_level_distribution(expressions, visualization_dir, args.case)

        print("\nDemo complete! Results are in:")
        print(f"  - Images: {image_dir}")
        print(f"  - Annotations: {annotation_dir}")
        print(f"  - Visualizations: {visualization_dir}")
        print(f"  - Reasoning level distribution: {os.path.join(visualization_dir, f'reasoning_levels_{args.case}.png')}")
    else:
        print(f"Error: Could not find the referring expressions file at {referring_expressions_path}")

if __name__ == "__main__":
    main()

"""
python demo_reasoning_levels.py --case extreme_simple_dfs --num_images 1000 --min_grid 2 --max_grid 8 --num_vis_samples 20 --seed 123 --output_dir rs2
"""