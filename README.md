# Reason-Synth: Synthetic Visual Data for Referring Expressions

A toolkit for generating synthetic images with geometric shapes and creating referring expressions for visual reasoning tasks.

## Overview

Reason-Synth generates synthetic images containing geometric shapes with various attributes, along with natural language referring expressions that identify specific objects. These datasets are ideal for training and evaluating models for visual reasoning, grounding, and referring expression comprehension.

## Features

- **Synthetic Image Generation**:
  - Various shapes (circle, triangle, square)
  - Multiple attributes (size, color, style, position)
  - Configurable grid layouts (from 2×2 to 8×8)
  - Clean annotations in JSONL format

- **Referring Expression Generation**:
  - Position-based expressions (DFS): "the object in the second row, third column"
  - Attribute-based expressions (BFS): "the large red triangle with a blue border"
  - Control over expression distribution and attributes
  - Support for "empty matches" (expressions that don't match any object)

- **Visualization Tools**:
  - Visualize expressions with highlighted target objects
  - Generate combined visualizations for presentations

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/reason-synth.git
cd reason-synth

# Set up environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete demo workflow:

```bash
python demo_workflow.py --output_dir demo_output --num_images 10
```

This will:
1. Generate 10 synthetic images with random shapes
2. Create referring expressions (both position-based and attribute-based)
3. Visualize the referring expressions with their target objects
4. Generate statistics about the dataset

## Customizing Expression Generation

You can control various aspects of the generated expressions:

```bash
python demo_workflow.py \
  --output_dir custom_demo \
  --num_images 20 \
  --min_grid 4 \
  --max_grid 6 \
  --sampling_dfs_ratio 0.6 \
  --sampling_existence_ratio 0.6 \
  --refexp_grid_min_row 1 \
  --refexp_grid_min_col 1 \
  --refexp_grid_max_row 5 \
  --refexp_grid_max_col 5 \
  --bfs_ratio_single_attr 0.3 \
  --bfs_ratio_two_attr 0.4 \
  --bfs_ratio_three_attr 0.2 \
  --bfs_ratio_four_attr 0.1 \
  --num_vis_samples 8
```

### Key Parameters:

- `sampling_dfs_ratio`: Controls the ratio of position-based (DFS) to attribute-based (BFS) expressions (0.6 = 60% DFS)
- `sampling_existence_ratio`: Controls the ratio of existence-based vs. random-based expressions (0.6 = 60% existence-based)
- `refexp_grid_min_row/col`, `refexp_grid_max_row/col`: Control the grid position range for DFS referring expressions
- `bfs_ratio_*`: Controls the complexity distribution of attribute-based expressions

## Understanding Referring Expressions

### Position-Based (DFS) Expressions

Position-based expressions refer to objects by their grid location. For example:
- "the shape in the first row, third column"
- "the object positioned second from the left in the last row"

### Attribute-Based (BFS) Expressions

Attribute-based expressions refer to objects by their visual properties. For example:
- "the blue circle"
- "the small shape with a border"
- "the large red triangle"

### Expression Sources

The system generates BFS expressions in two ways:
- **Existence-based**: Generated from attributes of objects that actually exist in the image
- **Random-based**: Generated from random attribute combinations, which may or may not exist in the image

Random-based expressions help create "empty match" cases where no objects match the expression, which is useful for training robust models.

## Outputs

After running the demo, you'll find:
- `images/`: The generated synthetic images
- `annotations/`:
  - `dataset.jsonl`: Image and object annotations
  - `referring_expressions.jsonl`: Generated referring expressions with matching objects
- `visualizations/`: Visualizations of expressions and their targets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
