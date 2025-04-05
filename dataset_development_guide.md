# Reason-Synth Dataset Development Guide

This document explains the process of generating and working with the Reason-Synth dataset, including sample generation and referring expression generation.

## 1. Sample Generation

The Reason-Synth dataset is composed of grid-based scenes with geometric shapes that have various attributes.

### Key Components

- **Grid Size**: Ranges from 2x2 to 8x8
- **Objects**: Geometric shapes placed in the grid
- **Object Attributes**:
  - `shape_type`: The geometric shape (e.g., triangle, square, circle)
  - `size`: Size of the shape (e.g., big, small)
  - `style`: Rendering style (solid, half, border)
  - `color1`: Primary color
  - `color2`: Secondary color (for half and border styles; for solid style, color2 is set to the same value as color1)
  - `grid_position`: [row, column] position in the grid
  - `position`: [x, y] pixel coordinates of the shape's center
  - `bbox`: Bounding box coordinates [left, top, right, bottom]
  - `rotation`: Rotation angle in degrees

### Data Structure

Based on the actual `large_dataset/annotations/dataset.jsonl` file, each line contains:

```json
{
  "image_path": "images/sample_0000.png",
  "scene": {
    "grid_size": [7, 7],
    "image_size": [800, 600],
    "region": [106, 71, 489, 471],
    "shapes": [
      {
        "shape_type": "triangle",
        "size": "big",
        "color1": "green",
        "color2": "yellow",
        "style": "border",
        "position": [150, 97],
        "rotation": 165.32960110771216,
        "bbox": [100, 51, 181, 132],
        "grid_position": [0, 0]
      },
      {
        "shape_type": "circle",
        "size": "small",
        "color1": "yellow",
        "color2": "yellow",  // Same as color1 for solid style
        "style": "solid",
        "position": [543, 104],
        "rotation": 0.0,  // Circle rotations are always 0
        "bbox": [523, 84, 563, 124],
        "grid_position": [0, 6]
      }
      // More shapes...
    ]
  }
}
```


### Sample Generation Script

The main script for generating samples is `generate_samples.py`. This script creates synthetic images and their corresponding annotations in JSONL format.

#### Key Parameters

- `--num-samples`: Number of samples to generate (default: 100)
- `--output-dir`: Output directory for dataset (default: 'data')
- `--seed`: Random seed for reproducibility (default: 42)
- `--sample`: Generate a single sample image for testing
- `--min-grid` / `--max-grid`: Minimum and maximum grid size (default: 2-3)
- `--min-region-ratio` / `--max-region-ratio`: Controls the area of the image used for the grid
- `--max-offset-ratio`: Maximum random offset as a ratio of cell size

#### Example Usage

```bash
# Generate 1000 samples with grid sizes from 2x2 to 5x5
python generate_samples.py --num-samples 1000 --min-grid 2 --max-grid 5 --output-dir dataset

# Generate a single sample image for testing
python generate_samples.py --sample --sample-output test_sample.png
```

#### Output

The script generates:
1. PNG images of synthetic scenes with shapes arranged in a grid
2. A JSONL annotation file (dataset.jsonl) containing detailed information about each scene
   - The annotation includes all object attributes including rotation angles
   - For solid style shapes, color2 is set to the same value as color1 (verified in the dataset)
   - Rotation for circles is always 0.0

## 2. Referring Expression Generation

Referring expressions are natural language phrases that identify one or more objects in a scene.

### Types of Referring Expressions

1. **DFS (Position-Based)**:
   - Identify objects by their grid position
   - Example: "the shape in the third row from the top, second column from the left"

2. **BFS (Attribute-Based)**:
   - Identify objects by their visual attributes
   - Example: "the large red triangle" or "the blue shape with an outline"

### Enhanced Parameters

#### DFS (Position-Based) Expressions
- **Grid Range Control**: Specify minimum and maximum grid positions for targeted referring expressions
  - `min_grid_row`, `min_grid_col`: Minimum row and column (0-indexed)
  - `max_grid_row`, `max_grid_col`: Maximum row and column (0-indexed)

#### BFS (Attribute-Based) Expressions
- **Specific Attribute Requirements**: Filter expressions to target objects with specific attributes
  - `shape_type`: Target specific shapes (e.g., "triangle", "circle", "square")
  - `size`: Target specific sizes (e.g., "big", "small")
  - `style`: Target specific styles (e.g., "solid", "half", "border")
  - `color1`: Target specific primary colors
  - `color2`: Target specific secondary colors (for "half" and "border" styles)

- **Proportional Category Sampling**: Control distribution across attribute complexity levels (default 1:1:1:1)
  - `single_attr_ratio`: Proportion of single attribute expressions (shape/color/size/style only)
  - `two_attr_ratio`: Proportion of two attribute combinations
  - `three_attr_ratio`: Proportion of three attribute combinations
  - `all_attr_ratio`: Proportion of all attribute expressions

- **Object Sampling Strategy** (default 1:1 ratio):
  - `sampling_strategy`:
    - "existence": First select an object, then create expressions from its properties
    - "random": Generate expressions with random attributes
    - "mixed": Use both strategies with equal probability (default)
  - Note: Generated expressions may match multiple objects or no objects, which is acceptable

**Example Usage in generate_referring_expressions.py**:
```python
# Generate BFS expressions with proportional category sampling
expressions = generator.generate_bfs_referring_expressions(
    n=100,
    single_attr_ratio=0.25,  # 25% single attribute expressions
    two_attr_ratio=0.25,     # 25% two attribute combinations
    three_attr_ratio=0.25,   # 25% three attribute combinations
    all_attr_ratio=0.25,     # 25% all attribute expressions
    sampling_strategy="mixed" # Use both existence and random sampling
)

# Generate BFS expressions with specific attribute focus
expressions = generator.generate_bfs_referring_expressions(
    n=50,
    sampling_strategy="existence", # Only use existence-based sampling
    specific_requirements={"shape_type": "triangle", "color1": "red"}
)
```

### Key Files

#### generate_referring_expressions.py

This file contains the `ReferringExpressionGenerator` class that generates referring expressions.

**Key Functions**:

- `generate_dfs_referring_expressions(n, min_grid, max_grid)`: Generates n position-based referring expressions
  - **Input**: Number of expressions to generate, minimum and maximum grid positions
  - **Output**: List of dictionaries with referring expressions and requirements

  ```python
  [
    {
      "referring_expression": "the shape in the third row from the top, second column from the left",
      "expression_type": "DFS",
      "target_requirements": {
        "row": 2,  # 0-indexed
        "column": 1  # 0-indexed
      }
    },
    // More expressions...
  ]
  ```

- `generate_bfs_referring_expressions(n, comprehensive_ratio, specific_requirements)`: Generates n attribute-based expressions
  - **Input**:
    - `n`: Number of expressions to generate
    - `comprehensive_ratio`: Ratio of comprehensive combinations to include
    - `specific_requirements`: Dictionary of specific attribute requirements
  - **Output**: List of dictionaries with referring expressions and requirements

  ```python
  [
    {
      "referring_expression": "the large red triangle",
      "expression_type": "BFS",
      "target_requirements": {
        "shape_type": "triangle",
        "size": "large",
        "color1": "red"
      }
    },
    // More expressions...
  ]
  ```

## 3. Creating the Dataset

The Reason-Synth dataset follows a two-stage process: first generating scenes with objects, then generating referring expressions that target those objects.

### Dataset Creation Process

1. **Scene Generation**:
   - Scenes are created with objects placed in grid cells
   - Each object has attributes like shape, size, color, and style
   - This produces the base dataset (dataset.jsonl)

2. **Referring Expression Generation**:
   - Unlike some datasets where expressions are created for specific target objects, Reason-Synth generates expressions first, then finds matching objects
   - This approach makes the dataset more challenging and realistic, as expressions may match multiple objects or no objects

3. **Matching Process**:
   - After generating expressions, the system searches for objects that match the requirements
   - Expressions that match no objects or multiple objects are acceptable for the dataset

### Key Files

#### create_referring_expressions_dataset.py

This file creates a dataset of referring expressions matched to objects in scenes.

**Key Functions**:

- `process_dataset(dataset_path)`: Processes the dataset and groups by grid size
  - **Input**: Path to dataset.jsonl
  - **Output**: Dictionary mapping grid sizes to lists of image data

- `matches_position_requirements(requirements, obj, grid_size)`: Checks if an object matches position requirements
  - **Input**: Requirements, object data, grid size
  - **Output**: Boolean indicating if object matches

- `matches_attribute_requirements(requirements, obj)`: Checks if an object matches attribute requirements
  - **Input**: Requirements, object data
  - **Output**: Boolean indicating if object matches

- `find_matching_objects(requirements, expression_type, objects, grid_size)`: Finds all objects matching requirements
  - **Input**: Requirements, expression type, objects list, grid size
  - **Output**: List of matching objects

- `generate_referring_expressions_dataset(input_path, output_path, dfs_ratio, max_images)`: Main function to generate dataset
  - **Input**:
    - `input_path`: Path to dataset.jsonl
    - `output_path`: Path to save referring expressions dataset
    - `dfs_ratio`: Ratio of DFS expressions (default: 0.75)
    - `max_images`: Maximum number of images to process
  - **Output**: JSONL file with referring expressions and matching objects

### Output Format

The referring expressions dataset is a JSONL file where each line contains:

```json
{
  "image_path": "images/sample_1234.png",
  "grid_size": [4, 4],
  "referring_expression": "the large red triangle",
  "expression_type": "BFS",
  "target_requirements": {
    "shape_type": "triangle",
    "size": "large",
    "color1": "red"
  },
  "primary_target_idx": -1,
  "matching_objects": [
    {
      "shape_type": "triangle",
      "size": "large",
      "color1": "red",
      "color2": "red",
      "style": "solid",
      "grid_position": [0, 1],
      "bbox": [x, y, width, height]
    },
    // More matching objects...
  ]
}
```

### Requirement Matching

The system uses two different matching functions depending on the expression type:

1. **Position-Based (DFS) Matching**:
   ```python
   def matches_position_requirements(requirements, obj, grid_size):
       # Get the object's grid position
       row, col = obj['grid_position']

       # Check if the position matches exactly
       return requirements['row'] == row and requirements['column'] == col
   ```

2. **Attribute-Based (BFS) Matching**:
   ```python
   def matches_attribute_requirements(requirements, obj):
       # Check shape type, size, style
       if 'shape_type' in requirements and requirements['shape_type'] != obj['shape_type']:
           return False
       # ... similar checks for size and style ...

       # Check colors based on style (solid, half, border)
       if obj['style'] == 'solid':
           # For solid shapes, color1 must match
           if 'color1' in requirements and requirements['color1'] != obj['color1']:
               return False
       elif obj['style'] == 'half':
           # For half shapes, must have both required colors in any order
           # ... color matching logic ...
       elif obj['style'] == 'border':
           # For border shapes, check main color and border color
           # ... color matching logic ...

       return True
   ```

### Generating vs. Targeting

A key aspect of the Reason-Synth dataset is that **expressions are generated independently of specific objects**:

1. **Random Generation Approach**:
   - Expressions are generated based on templates and random attribute selections
   - Grid positions (for DFS) and attribute requirements (for BFS) are chosen randomly
   - If specific requirements are provided, they constrain the random selection

2. **Finding Matches After Generation**:
   - After generating an expression, the system searches for matching objects
   - This means one expression might match multiple objects (ambiguous) or no objects (invalid)
   - Expressions that match no objects or multiple objects are acceptable for the dataset

3. **Implications**:
   - This approach mimics real-world referring: people don't always generate perfectly unambiguous references
   - It creates a more challenging dataset where models must handle ambiguity
   - It allows for studying how different types of expressions resolve to different numbers of objects

### Workflow for Creating the Dataset

```
                  ┌─────────────────┐
                  │ Generate Scenes │
                  └────────┬────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ dataset.jsonl     │
                 │ (scenes & objects)│
                 └────────┬──────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ Generate Referring Expressions   │
        │ - DFS (position-based)           │
        │ - BFS (attribute-based)          │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │ Match Expressions to Objects     │
        │ - matches_position_requirements  │
        │ - matches_attribute_requirements │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │ refer_exp_dataset.jsonl         │
        │ (expressions + matching objects) │
        └─────────────────────────────────┘
```

### Visualization and Validation

To ensure the dataset quality, we've created a visualization tool that displays the referring expressions along with their matching objects:

```bash
# Generate a small test dataset and visualize it
python visualize_referring_expressions.py --generate_test --test_dir visualization_test

# Visualize an existing dataset
python visualize_referring_expressions.py --dataset test_dataset/annotations/refer_exp_dataset.jsonl --image_dir test_dataset/images
```

This visualization helps verify that:
1. Expressions match the correct objects
2. Grid position references (DFS) are accurate
3. Attribute references (BFS) correctly identify objects with the specified attributes

## Development Pipeline

1. **Configure Objects**: Define shapes, sizes, colors, and styles in `object_config.json`
2. **Generate Samples**: Create grid-based scenes with objects and save to `dataset.jsonl`
3. **Generate Referring Expressions**:
   - Run `generate_referring_expressions.py` to create expression templates
   - Run `create_referring_expressions_dataset.py` to match expressions to objects
4. **Validate Dataset**: Ensure expressions match the intended objects
5. **Use Dataset**: Train and evaluate models using the generated dataset

### Command Line Usage

```bash
# Generate a dataset with position-based expressions for a specific grid range
python create_referring_expressions_dataset.py \
  --input large_dataset/annotations/dataset.jsonl \
  --output large_dataset/annotations/refer_exp_dataset.jsonl \
  --min_grid_row 1 --min_grid_col 1 \
  --max_grid_row 5 --max_grid_col 5

# Generate a dataset with attribute-based expressions for specific objects
python create_referring_expressions_dataset.py \
  --input large_dataset/annotations/dataset.jsonl \
  --output large_dataset/annotations/refer_exp_dataset.jsonl \
  --shape_type triangle --color1 red --style border

# Generate a dataset with proportional BFS category sampling
python create_referring_expressions_dataset.py \
  --input dataset.jsonl \
  --output refer_exp_dataset.jsonl \
  --dfs_ratio 0.0 \
  --sampling_strategy mixed \
  --single_attr_ratio 0.25 \
  --two_attr_ratio 0.25 \
  --three_attr_ratio 0.25 \
  --all_attr_ratio 0.25

# Generate a combined dataset with both types of constraints
python create_referring_expressions_dataset.py \
  --input large_dataset/annotations/dataset.jsonl \
  --output large_dataset/annotations/refer_exp_dataset.jsonl \
  --dfs_ratio 0.5 \
  --min_grid_row 1 --max_grid_row 5 \
  --shape_type circle --style solid --color1 blue
```

### Testing and Validation

A dedicated test script (`test_referring_expressions.py`) is available to test the referring expression generation:

```bash
# Test DFS expressions with grid range constraints
python test_referring_expressions.py --mode dfs --max_grid_row 5 --max_grid_col 5

# Test BFS expressions with specific attribute requirements
python test_referring_expressions.py --mode bfs --shape_type triangle --color1 red

# Generate and validate a test dataset
python test_referring_expressions.py --mode dataset \
  --input dataset.jsonl --output test_output.jsonl \
  --min_grid_row 1 --max_grid_row 4 \
  --shape_type circle --style border
```

This pipeline allows for flexible generation of diverse referring expressions that can be used to train and evaluate models for visual reasoning tasks.