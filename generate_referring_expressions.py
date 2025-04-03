#!/usr/bin/env python
"""
Generate templates for DFS (position-based) and BFS (attribute-based) referring expressions for the reason-synth dataset.

What are DFS and BFS?
- DFS (Depth-First Search) refers to position-based expressions where objects are identified by their grid position
  (e.g., "the shape in the third row, second column")
- BFS (Breadth-First Search) refers to attribute-based expressions where objects are identified by their visual properties
  (e.g., "the large red triangle" or "the blue shape with an outline")

This metaphor represents different ways people might refer to objects:
- DFS (Depth-First Search) refers to a systematic approach where we follow a clear path to identify an object, like listing items one by one until we reach the target (position-based references are a specific case of this)
- BFS (Breadth-First Search) refers to identifying objects by considering all possible options and carefully examining their distinctive visual features to decide which one matches

IMPORTANT REQUIREMENTS:
1. All referring expressions must use natural language that sounds like a human would ask
2. Avoid technical terminology like coordinates (x,y)
3. Use specific descriptive language for styles instead of our "style" terms (e.g. "solid" -> "filled", "half" -> "split into two colors", "border" -> "outlined")
4. Referring expressions should be diverse and cover various ways to ask about objects
5. Remove question phrases like "find" or "where is" - just keep the expression itself
"""

import json
import random
import itertools
from typing import List, Dict, Tuple, Optional

# Load the object configuration
CONFIG_PATH = "src/object_config.json"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

SHAPE_TYPES = CONFIG["shape_types"]
SIZES = CONFIG["sizes"]
COLORS = list(CONFIG["colors"].keys())
STYLES = CONFIG["styles"]

# Style descriptions that are more natural than just "style"
STYLE_DESCRIPTIONS = {
    "solid": ["filled", "completely {color}", "solid-colored", "uniform"],
    "half": ["split into two colors", "two-toned", "half-and-half", "divided into two parts"],
    "border": ["outlined", "with an outline", "with a border", "framed"]
}

# Ordinal number mapping for natural language (used only for DFS/position-based expressions)
ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"
}

# Direction specifications
DIRECTION_SPECS = {
    "row_from_top": ["from the top", "counting from top to bottom", "starting from the top", "from the top down"],
    "row_from_bottom": ["from the bottom", "counting from bottom to top", "starting from the bottom", "from the bottom up"],
    "col_from_left": ["from the left", "counting from left to right", "starting from the left", "from left side"],
    "col_from_right": ["from the right", "counting from right to left", "starting from the right", "from right side"]
}

class ReferringExpressionGenerator:
    def __init__(self, max_grid_size: Tuple[int, int] = (8, 7)):
        """
        Initialize the referring expression generator with the maximum grid size.
        Args:
            max_grid_size: Maximum grid size (rows, columns) in the dataset
        """
        self.max_rows, self.max_cols = max_grid_size
        self.dfs_templates = []
        self.bfs_templates = []
        self.bfs_template_patterns = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all templates for both DFS and BFS referring expressions."""
        # DFS (position-based) templates as referring expressions
        self.dfs_templates = [
            # Standard templates with clear direction
            "the object in the {row_num} row {row_dir}, {col_num} column {col_dir}",
            "the object in the {ordinal_row} row {row_dir}, {ordinal_col} column {col_dir}",
            "the {ordinal_col} object {col_dir} in the {ordinal_row} row {row_dir}",
            "the object located in the {ordinal_row} row {row_dir}, {ordinal_col} position {col_dir}",
            "the shape in the {ordinal_row} row {row_dir} at position {ordinal_col} {col_dir}",
            "the object in the {ordinal_row} row {row_dir} and {ordinal_col} column {col_dir}",
            "the shape in the {row_num} row {row_dir}, {col_num} column {col_dir}",
            "the shape in the {ordinal_row} row {row_dir} at the {ordinal_col} spot {col_dir}",
            "the shape in the {ordinal_row} row {row_dir}, {ordinal_col} spot {col_dir}",

            # Different ordering of words
            "the {ordinal_col} item {col_dir} on the {ordinal_row} row {row_dir}",
            "on row {row_num} {row_dir}, the {col_num} object {col_dir}",
            "row {row_num} {row_dir}, column {col_num} {col_dir}",
            "{ordinal_col} from {col_alt_dir} on the {ordinal_row} row {row_dir}",
            "{ordinal_row} row {row_dir}, {ordinal_col} column {col_dir} - that object",
            "the {ordinal_col} shape {col_dir} of row {row_num} {row_dir}",

            # More colloquial or alternative ways to describe positions
            "the {ordinal_col} thing {col_dir} on the {ordinal_row} row {row_dir}",
            "the object at {col_num} across {col_dir}, {row_num} down {row_dir}",
            "the shape positioned {ordinal_col} {col_dir} in row number {row_num} {row_dir}",
            "the thing in row {row_num} {row_dir}, spot {col_num} {col_dir}",
            "the {ordinal_col} thing {col_dir} in the {ordinal_row} line {row_dir}",

            # Directional counting
            "starting from the {col_alt_dir}, the {ordinal_col} object in the {ordinal_row} row {row_dir}",
            "counting from the {row_alt_dir}, the shape in row {ordinal_row}, column {ordinal_col} {col_dir}",
            "if you start from the {col_alt_dir}, the {ordinal_col} shape in row {row_num} {row_dir}",
            "the {ordinal_col} object when counting from {col_alt_dir} in the {ordinal_row} row {row_dir}"
        ]

        # BFS templates organized by attribute patterns
        self.bfs_template_patterns = {
            # Single attribute templates
            "shape_only": ["the {shape_type}"],
            "color_only": ["the {color} object", "the {color} shape"],
            "size_only": ["the {size} shape", "the {size} object"],
            "style_only": ["the {style_desc} object", "the {style_desc} shape"],

            # Two attribute combinations
            "shape_color": ["the {color} {shape_type}"],
            "shape_size": ["the {size} {shape_type}"],
            "shape_style": ["the {style_desc} {shape_type}"],
            "color_size": ["the {color} {size} object", "the {size} {color} shape"],
            "color_style": ["the {color} {style_desc} shape", "the {style_desc} {color} object"],
            "size_style": ["the {size} {style_desc} object", "the {style_desc} {size} shape"],

            # Three attribute combinations
            "shape_color_size": ["the {size} {color} {shape_type}", "the {color} {size} {shape_type}"],
            "shape_color_style": ["the {color} {style_desc} {shape_type}", "the {style_desc} {color} {shape_type}"],
            "shape_size_style": ["the {size} {style_desc} {shape_type}", "the {style_desc} {size} {shape_type}"],
            "color_size_style": ["the {size} {color} {style_desc} object", "the {color} {style_desc} {size} shape"],

            # All attributes
            "all_attributes": ["the {size} {color} {style_desc} {shape_type}"],

            # Style-specific templates (solid)
            "solid_style": [
                "the completely filled {color} {shape_type}",
                "the solid {color} {size} shape",
                "the uniform {color} {shape_type}",
                "the {size} {shape_type} filled with {color}"
            ],

            # Style-specific templates (half)
            "half_style": [
                "the object that's half {color1} and half {color2}",
                "the {shape_type} that's split between {color1} and {color2}",
                "the {size} shape that's divided into {color1} and {color2} parts",
                "the {size} {shape_type} that has both {color1} and {color2} in it",
                "the two-toned {size} {shape_type}",
                "the half-and-half {color1} and {color2} {shape_type}"
            ],

            # Style-specific templates (border)
            "border_style": [
                "the object with a {color} outline",
                "the {shape_type} that has a {color} border around it",
                "the {size} {shape_type} outlined in {color}",
                "the {color1} {shape_type} that has a {color2} border around it",
                "the framed {color} {shape_type}",
                "the {size} {shape_type} with a {color} edge"
            ],

            # Multiple objects
            "multiple_objects": [
                "all {color} objects",
                "all {size} {shape_type}s",
                "all {style_desc} objects",
                "all {color} {size} shapes",
                "all {color} {style_desc} objects",
                "all {size} {style_desc} shapes",
                "all {shape_type}s that are either {color1} or {color2}",
                "all {shape_type}s with {color} in them"
            ],

            # Comprehensive combinations (handled by function)
            "comprehensive": None  # Will use generate_all_attribute_combinations
        }

        # Flatten templates for backward compatibility
        self.bfs_templates = []
        for key, templates in self.bfs_template_patterns.items():
            if templates:  # Skip the None value for "comprehensive"
                self.bfs_templates.extend(templates)

    def generate_dfs_referring_expressions(self, n: int = 10) -> List[str]:
        """
        Generate a specified number of DFS (position-based) referring expressions.
        Args:
            n: Number of referring expressions to generate
        Returns:
            List of generated referring expressions
        """
        referring_expressions = []
        for _ in range(n):
            template = random.choice(self.dfs_templates)
            row = random.randint(1, self.max_rows)
            col = random.randint(1, self.max_cols)

            # Get ordinals if needed
            ordinal_row = ORDINALS.get(row, f"{row}th")
            ordinal_col = ORDINALS.get(col, f"{col}th")

            # Select random direction specifications
            row_dir = random.choice(DIRECTION_SPECS["row_from_top"])
            row_alt_dir = random.choice(DIRECTION_SPECS["row_from_bottom"])
            col_dir = random.choice(DIRECTION_SPECS["col_from_left"])
            col_alt_dir = random.choice(DIRECTION_SPECS["col_from_right"])

            # Format the template
            referring_expression = template.format(
                row=row, col=col,
                row_num=f"{row}{'st' if row == 1 else 'nd' if row == 2 else 'rd' if row == 3 else 'th'}",
                col_num=f"{col}{'st' if col == 1 else 'nd' if col == 2 else 'rd' if col == 3 else 'th'}",
                ordinal_row=ordinal_row,
                ordinal_col=ordinal_col,
                row_dir=row_dir,
                row_alt_dir=row_alt_dir,
                col_dir=col_dir,
                col_alt_dir=col_alt_dir
            )
            referring_expressions.append(referring_expression)

        return referring_expressions

    def generate_bfs_referring_expressions(self, n: int = 10, comprehensive_ratio: float = 0.15) -> List[str]:
        """
        Generate a specified number of BFS (attribute-based) referring expressions.
        Args:
            n: Number of referring expressions to generate
        Returns:
            List of generated referring expressions
        """
        referring_expressions = []

        # Create all combinations of attributes for more complex referring expressions
        shape_color_combos = list(itertools.product(SHAPE_TYPES, COLORS))
        size_shape_combos = list(itertools.product(SIZES, SHAPE_TYPES))
        color_pairs = list(itertools.combinations(COLORS, 2))

        # Get list of all pattern keys (excluding "comprehensive")
        pattern_keys = [key for key in self.bfs_template_patterns.keys() if key != "comprehensive"]

        # Allocate some expressions for comprehensive combinations
        num_comprehensive = max(1, int(n * comprehensive_ratio))
        num_from_patterns = n - num_comprehensive

        # First, add comprehensive combinations
        if num_comprehensive > 0:
            all_combinations = self.generate_all_attribute_combinations()
            comprehensive_samples = random.sample(all_combinations, min(num_comprehensive, len(all_combinations)))
            referring_expressions.extend(comprehensive_samples)

        # Then, add expressions from patterns
        if num_from_patterns > 0:
            # Distribute expressions across pattern types
            patterns_per_category = {}
            remaining = num_from_patterns

            # First pass: ensure at least one of each pattern if possible
            for key in pattern_keys:
                if remaining > 0:
                    patterns_per_category[key] = 1
                    remaining -= 1
                else:
                    patterns_per_category[key] = 0

            # Second pass: distribute remaining expressions proportionally
            if remaining > 0:
                for _ in range(remaining):
                    key = random.choice(pattern_keys)
                    patterns_per_category[key] += 1

            # Generate expressions based on the distribution
            for key, count in patterns_per_category.items():
                if count == 0:
                    continue

                templates = self.bfs_template_patterns[key]

                for _ in range(count):
                    template = random.choice(templates)

                    # Select random attributes
                    shape_type = random.choice(SHAPE_TYPES)
                    color = random.choice(COLORS)
                    size = random.choice(SIZES)
                    style = random.choice(STYLES)

                    # Get a natural description for the selected style
                    if style in STYLE_DESCRIPTIONS:
                        style_desc = random.choice(STYLE_DESCRIPTIONS[style])
                    else:
                        style_desc = style  # Fallback

                    # For "half_style" templates that need two colors
                    if key == "half_style":
                        color1, color2 = random.choice(color_pairs)
                        referring_expression = template.format(
                            shape_type=shape_type,
                            size=size,
                            color1=color1,
                            color2=color2,
                            style_desc=style_desc
                        )
                    # For "border_style" templates
                    elif key == "border_style":
                        color1 = random.choice(COLORS)
                        color2 = random.choice([c for c in COLORS if c != color1])
                        referring_expression = template.format(
                            shape_type=shape_type,
                            size=size,
                            color=color,
                            color1=color1,
                            color2=color2,
                            style_desc=style_desc
                        )
                    # For other templates (standard attribute templates)
                    else:
                        referring_expression = template.format(
                            shape_type=shape_type,
                            color=color,
                            size=size,
                            style_desc=style_desc.format(color=color) if "{color}" in style_desc else style_desc,
                            color1=random.choice(COLORS),
                            color2=random.choice([c for c in COLORS if c != color])
                        )

                    referring_expressions.append(referring_expression)

        # Shuffle the results to mix comprehensive and pattern-based expressions
        random.shuffle(referring_expressions)

        # Return only the requested number
        return referring_expressions[:n]

    def generate_all_attribute_combinations(self) -> List[str]:
        """
        Generate referring expressions covering all possible attribute combinations.
        Returns:
            List of referring expressions with all attribute combinations
        """
        referring_expressions = []

        # Generate comprehensive attribute combinations
        for shape, size, style in itertools.product(SHAPE_TYPES, SIZES, STYLES):
            # Handle solid style
            if style == "solid":
                for color in COLORS:
                    referring_expressions.append(f"the {size} {color} solid {shape}")
                    referring_expressions.append(f"the completely filled {size} {color} {shape}")
                    referring_expressions.append(f"the uniform {color} {size} {shape}")

            # Handle half style
            elif style == "half":
                for color1, color2 in itertools.combinations(COLORS, 2):
                    referring_expressions.append(f"the {size} {shape} that's half {color1} and half {color2}")
                    referring_expressions.append(f"the two-toned {size} {shape} in {color1} and {color2}")
                    referring_expressions.append(f"the {size} {shape} divided between {color1} and {color2}")

            # Handle border style
            elif style == "border":
                for main_color, border_color in itertools.product(COLORS, COLORS):
                    if main_color != border_color:
                        referring_expressions.append(f"the {size} {main_color} {shape} with a {border_color} outline")
                        referring_expressions.append(f"the {main_color} {size} {shape} bordered in {border_color}")
                        referring_expressions.append(f"the {size} {shape} that's {main_color} with a {border_color} frame")

        # Add additional combinations with DFS patterns (for completeness in the dataset)
        for shape, color, size in itertools.product(SHAPE_TYPES, COLORS, SIZES):
            referring_expressions.append(f"the {color} {size} {shape}")
            referring_expressions.append(f"the {size} {shape} in {color}")
            referring_expressions.append(f"the {color} {shape} of {size} size")

        return referring_expressions

def main():
    """Generate and print referring expression templates."""
    generator = ReferringExpressionGenerator()

    print("=== DFS (Position-Based) Referring Expressions ===")
    dfs_referring_expressions = generator.generate_dfs_referring_expressions(10)
    for i, referring_expression in enumerate(dfs_referring_expressions, 1):
        print(f"{i}. {referring_expression}")

    print("\n=== BFS (Attribute-Based) Referring Expressions ===")
    bfs_referring_expressions = generator.generate_bfs_referring_expressions(10)
    for i, referring_expression in enumerate(bfs_referring_expressions, 1):
        print(f"{i}. {referring_expression}")
    print("\n=== All Attribute Combinations (Sample) ===")
    all_combinations = generator.generate_all_attribute_combinations()
    # Print just a sample of these combinations
    for i, referring_expression in enumerate(random.sample(all_combinations, min(10, len(all_combinations))), 1):
        print(f"{i}. {referring_expression}")

    print(f"\nTotal templates generated:")
    print(f"- DFS referring expression templates: {len(generator.dfs_templates)}")
    print(f"- BFS referring expression templates: {len(generator.bfs_templates)}")
    print(f"- BFS template patterns: {len(generator.bfs_template_patterns)}")
    print(f"- All attribute combinations: {len(all_combinations)}")

    # Print the distribution of templates by pattern
    print("\nBFS Template Pattern Distribution:")
    for key, templates in generator.bfs_template_patterns.items():
        if templates:
            print(f"- {key}: {len(templates)} templates")
        else:
            print(f"- {key}: Using function")

if __name__ == "__main__":
    main()
