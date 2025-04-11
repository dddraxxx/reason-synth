#!/usr/bin/env python
"""
Script to generate synthetic images and annotations for referring expressions.

Arguments:
    --num-samples: Number of samples to generate (default: 100).

    --output-dir: Output directory for dataset (default: 'data').

    --seed: Random seed for reproducibility (default: 42).

    --sample: Generate a single sample image for testing.

    --sample-output: Output path for sample image (default: 'sample.png').

    --cell-size: Base size of a grid cell in pixels (default: 80). Used with grid size
        to calculate the base placement region size.

    --region-variation-ratio: Controls random scaling of the placement region. The region
        size, calculated from cell-size and grid-size, is randomly scaled by a factor
        between 1-ratio and 1+ratio (default: 0.1).

    --max-offset-ratio: Controls randomness in object positioning. Objects are placed
        on a regular grid, then randomly offset by up to max_offset_ratio*cell_size
        in each direction. Higher values create more irregular patterns (default: 0.2).

    --max-overlap-ratio: Maximum allowed overlap distance ratio (relative to smaller bbox diagonal).
        0=touching, >0=overlap (default: 0).

    --min-grid/--max-grid: Defines the range of grid dimensions (rows and columns).
        The generator will randomly choose a grid size between these bounds.
        For example, with min_grid=2, max_grid=3, possible grids are 2x2, 2x3, 3x2, 3x3.
        More objects means smaller individual objects (defaults: 2-3).

Coordinates are calculated as follows:
    1. Base region size is calculated: rows*cell_size, cols*cell_size
    2. Region size is randomly scaled using region-variation-ratio
    3. Region is centered within the image (clamped if needed)
    4. Actual cell dimensions are derived from the final region size
    5. A grid of regular cells is created within the region
    6. Each cell's base position is at its center
    7. Random offsets are applied to each position (controlled by max_offset_ratio)
    8. Object size is determined by available cell size
    9. Distance between objects depends on grid cell size and random offsets
"""

import argparse
import os
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
    parser.add_argument('--cell-size', '-cs', type=int, default=120,
                        help='Base size of a grid cell in pixels')
    parser.add_argument('--region-variation-ratio', '-rv', type=float, default=0.1,
                        help='Random variation (+/-) ratio for the placement region size')
    parser.add_argument('--max-offset-ratio', '-mo', type=float, default=0.2,
                        help='Maximum offset as a ratio of cell size')
    parser.add_argument('--max-overlap-ratio', '-ol', type=float, default=0,
                        help='Maximum allowed overlap distance ratio (relative to smaller bbox diagonal). 0=touching, >0=overlap.')
    parser.add_argument('--min-grid', '-mig', type=int, default=2,
                        help='Minimum grid size (both rows and columns)')
    parser.add_argument('--max-grid', '-mag', type=int, default=3,
                        help='Maximum grid size (both rows and columns)')
    parser.add_argument('--verbose-placement', '-vp', action='store_true',
                        help='Enable verbose output for shape placement attempts')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set debugging environment variable if verbose placement is enabled
    if args.verbose_placement:
        os.environ['dp'] = '1'
        print("Verbose placement reporting enabled")

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
            cell_size=args.cell_size,
            region_variation_ratio=args.region_variation_ratio,
            max_offset_ratio=args.max_offset_ratio,
            max_overlap_ratio=args.max_overlap_ratio,
            seed=args.seed
        )
        print("Done!")

if __name__ == "__main__":
    main()