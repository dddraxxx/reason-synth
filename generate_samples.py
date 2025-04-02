#!/usr/bin/env python
"""
Script to generate synthetic images and annotations for referring expressions.
"""

import argparse
from src.generator import generate_dataset, generate_sample_image

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic images and annotations')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sample', action='store_true',
                        help='Generate a single sample image for testing')
    parser.add_argument('--sample-output', type=str, default='sample.png',
                        help='Output path for sample image')
    parser.add_argument('--min-region-ratio', type=float, default=0.5,
                        help='Minimum ratio of image size to use for grid placement')
    parser.add_argument('--max-region-ratio', type=float, default=0.8,
                        help='Maximum ratio of image size to use for grid placement')
    parser.add_argument('--max-offset-ratio', type=float, default=0.2,
                        help='Maximum offset as a ratio of cell size')
    parser.add_argument('--min-grid', type=int, default=2,
                        help='Minimum grid size (both rows and columns)')
    parser.add_argument('--max-grid', type=int, default=3,
                        help='Maximum grid size (both rows and columns)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Define region ratio range
    region_ratio_range = (args.min_region_ratio, args.max_region_ratio)

    # Generate grid sizes based on min_grid and max_grid
    grid_sizes = []
    for rows in range(args.min_grid, args.max_grid + 1):
        for cols in range(args.min_grid, args.max_grid + 1):
            grid_sizes.append((rows, cols))

    if args.sample:
        print("Generating a single sample image...")
        generate_sample_image(args.sample_output)
    else:
        print(f"Generating {args.num_samples} samples...")
        print(f"Using grid sizes: {grid_sizes}")
        generate_dataset(
            num_samples=args.num_samples,
            grid_sizes=grid_sizes,
            output_dir=args.output_dir,
            region_ratio_range=region_ratio_range,
            max_offset_ratio=args.max_offset_ratio,
            seed=args.seed
        )
        print("Done!")

if __name__ == "__main__":
    main()