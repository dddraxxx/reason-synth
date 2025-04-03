# Reason-Synth Development Guide

This document provides a comprehensive overview of the Reason-Synth project, explaining each component, terminology, and the overall workflow to help developers understand and contribute to the project.

## Project Overview

Reason-Synth generates synthetic visual data specifically designed for referring expression tasks in computer vision and visual reasoning. It creates images with geometric shapes of various properties arranged in grid layouts, along with corresponding natural language expressions that refer to specific objects in the scene.

## Development Guidelines

### General Rules

1. **Keep it simple**: Avoid unnecessary complexity in the codebase.
2. **One function, one purpose**: Each function should have a single, well-defined purpose.
3. **Clean interfaces**: Maintain clean interfaces between components.
4. **Code organization**: Keep the directory structure clean and organized.

### Documentation Rules

1. **Complex File Documentation**: For complex core files, include a section at the top of the file that explains:
   - The core logic and algorithms used
   - The main design decisions and their rationale
   - How the file interacts with other components
   - Any non-obvious behaviors or edge cases to be aware of

This approach helps new developers understand the code's purpose and design philosophy without having to reverse-engineer the implementation details.

## Key Terminology

- **Referring Expression**: A natural language phrase that uniquely identifies an object in a scene, such as "the red triangle" or "the shape in the top-left corner".
- **DFS (Depth-First Search)**: In this project, DFS refers to position-based referring expressions that identify objects by their grid location.
- **BFS (Breadth-First Search)**: In this project, BFS refers to attribute-based referring expressions that identify objects by their visual properties.
- **Bounding Box (bbox)**: The smallest rectangle that completely encloses a shape.

## Core Components

### 1. Shape Generation (`src/shapes.py`)

This module defines the `Shape` class and implements drawing functions for various geometric shapes.

**Key Features:**
- Shapes can be triangles, squares, or circles
- Each shape has size, color, style, and rotation properties
- Special handling for rotated shapes with proper bounding box calculation
- Support for different styles: solid, half (two-color), and border (outlined)

**Example usage:**
```python
triangle = Shape("triangle", "big", "red", style="solid", rotation=30)
```

### 2. Scene Composition (`src/scene.py`)

The `Scene` class handles the arrangement of shapes on a grid within an image.

**Key Features:**
- Configurable grid sizes
- Random region placement within the image
- Minimum spacing between objects based on bounding boxes
- Object collision detection
- JSONL annotation format export

**Example usage:**
```python
scene = Scene(grid_size=(3, 3))
scene.add_shape(shape, (0, 1))  # Add shape to row 0, column 1
```

### 3. Dataset Generation (`src/generator.py`)

This module provides functions for generating complete datasets.

**Key Features:**
- Generate individual random shapes
- Generate complete scenes with varied grid sizes
- Save images and annotations to specified directories
- Support for configurable parameters (seed, region ratio, offset ratio)

**Example usage:**
```python
generate_dataset(num_samples=100, grid_sizes=[(2, 2), (3, 3), (4, 4)])
```

### 4. Referring Expression Generation (`generate_referring_expressions.py`)

This script generates natural language referring expressions for objects in scenes.

**Key Features:**
- Two approaches to referring expressions:
  - Position-based (DFS): "the shape in the third row, second column"
  - Attribute-based (BFS): "the large red triangle"
- Template-based generation for natural-sounding language
- Support for all possible attribute combinations
- Style-specific templates for different shape styles

**Example usage:**
```python
generator = ReferringExpressionGenerator()
dfs_expressions = generator.generate_dfs_referring_expressions(10)
bfs_expressions = generator.generate_bfs_referring_expressions(10)
```

## Script Descriptions

### `generate_samples.py`

Creates a dataset of synthetic images with annotations.

**Key Parameters:**
- `--num-samples`: Number of samples to generate
- `--output-dir`: Directory to save the dataset
- `--min-grid` and `--max-grid`: Min/max grid sizes
- `--seed`: Random seed for reproducibility

### `generate_gallery.py`

Creates gallery images showing all possible shape combinations.

**Key Parameters:**
- `--output`: Output path for standard gallery
- `--rotated-output`: Output path for gallery with rotated shapes

### `generate_referring_expressions.py`

Generates templates for DFS (position-based) and BFS (attribute-based) referring expressions.

**Output Examples:**
- DFS: "the object in the third row, second column"
- BFS: "the big red triangle"

### `visualize.py`

Visualizes images with annotations, including bounding boxes and grid regions.

**Key Parameters:**
- `--image`: Path to the image
- `--jsonl`: Path to the annotation file
- `--show-grid`, `--show-bbox`, `--show-region`: Visualization options

## Workflow

The typical workflow for using this project is:

1. **Generate Dataset**: Use `generate_samples.py` to create a dataset of images with shapes
2. **Generate Referring Expressions**: Use `generate_referring_expressions.py` to create referring expressions
3. **Visualize Results**: Use `visualize.py` to view the generated images with annotations

## File Structure

```
reason-synth/
├── src/                             # Core code
│   ├── shapes.py                    # Shape classes and drawing
│   ├── scene.py                     # Scene composition
│   ├── generator.py                 # Dataset generation
│   └── object_config.json           # Shape attributes
├── generate_samples.py              # Sample generation script
├── generate_gallery.py              # Gallery generation script
├── generate_referring_expressions.py # Referring expression generation
├── visualize.py                     # Visualization tool
├── gallery.png                      # Shape gallery
├── gallery_rotated.png              # Rotated shape gallery
├── README.md                        # Project overview
└── DEVELOPMENT_GUIDE.md             # This file
```

## Advanced Concepts

### Grid Positioning System

Objects are placed in a grid with random offsets within each cell. The grid is placed within a random region of the image, and the size of this region is randomly determined within configurable limits.

### Minimum Spacing Algorithm

To ensure objects don't overlap or appear too close, a minimum spacing algorithm is used. This algorithm:
1. Calculates the bounding box for each shape, accounting for rotation
2. Determines the minimum required distance based on the smaller shape's bounding box
3. Validates positions to ensure the minimum distance is maintained

### Referring Expression Templates

Referring expressions are generated using templates with placeholders for attributes:
- Simple templates: "the {color} {shape_type}"
- Complex templates: "the {size} {color} {style_desc} {shape_type}"

The placeholders are filled with appropriate values to create natural-sounding expressions.

## Use Cases

This project can be used for:
- Training and evaluating vision-language models
- Testing object recognition systems
- Creating benchmarks for visual reasoning
- Developing educational materials for computer vision concepts
