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
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all templates for both DFS and BFS referring expressions."""
        # DFS (position-based) templates as referring expressions
        self.dfs_templates = [
            "the object in the {row_num} row, {col_num} column",
            "the object in the {ordinal_row} row, {ordinal_col} column",
            "the {ordinal_col} object from the left in the {ordinal_row} row from the top",
            "the object located in the {ordinal_row} row, {ordinal_col} position",
            "the shape in the {ordinal_row} row at position {ordinal_col}",
            "the object in the {ordinal_row} row and {ordinal_col} column",
            "the shape in the {row_num} row, {col_num} column",
            "the shape in the {ordinal_row} row at the {ordinal_col} spot",
            "the object in the {ordinal_row} row, {ordinal_col} column",
            "the shape in the {ordinal_row} row from the top, {ordinal_col} column from the left",
            "the shape in the {ordinal_row} row, {ordinal_col} spot"
        ]
        # BFS (attribute-based) templates as referring expressions
        self.bfs_templates = [
            # Single attribute templates
            "the {shape_type}",
            "the {color} object",
            "the {size} shape",
            "the {style_desc} object",

            # Two attribute combinations
            "the {color} {shape_type}",
            "the {size} {shape_type}",
            "the {style_desc} {shape_type}",
            "the {color} {size} object",
            "the {color} {style_desc} shape",
            "the {size} {style_desc} object",

            # Three attribute combinations
            "the {size} {color} {shape_type}",
            "the {color} {style_desc} {shape_type}",
            "the {size} {style_desc} {shape_type}",
            "the {size} {color} {style_desc} object",

            # Four attribute combination (all attributes)
            "the {size} {color} {style_desc} {shape_type}",

            # Style-specific templates (solid)
            "the completely filled {color} {shape_type}",
            "the solid {color} {size} shape",
            "the uniform {color} {shape_type}",
            "the {size} {shape_type} filled with {color}",

            # Style-specific templates (half)
            "the object that's half {color1} and half {color2}",
            "the {shape_type} that's split between {color1} and {color2}",
            "the {size} shape that's divided into {color1} and {color2} parts",
            "the {size} {shape_type} that has both {color1} and {color2} in it",
            "the two-toned {size} {shape_type}",
            "the half-and-half {color1} and {color2} {shape_type}",

            # Style-specific templates (border)
            "the object with a {color} outline",
            "the {shape_type} that has a {color} border around it",
            "the {size} {shape_type} outlined in {color}",
            "the {color1} {shape_type} that has a {color2} border around it",
            "the framed {color} {shape_type}",
            "the {size} {shape_type} with a {color} edge",

            # Multiple objects (keep "all" for these)
            "all {color} objects",
            "all {size} {shape_type}s",
            "all {style_desc} objects",
            "all {color} {size} shapes",
            "all {color} {style_desc} objects",
            "all {size} {style_desc} shapes",
            "all {shape_type}s that are either {color1} or {color2}",
            "all {shape_type}s with {color} in them"
        ]
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

            # Format the template
            referring_expression = template.format(
                row=row, col=col,
                row_num=f"{row}{'st' if row == 1 else 'nd' if row == 2 else 'rd' if row == 3 else 'th'}",
                col_num=f"{col}{'st' if col == 1 else 'nd' if col == 2 else 'rd' if col == 3 else 'th'}",
                ordinal_row=ordinal_row,
                ordinal_col=ordinal_col
            )
            referring_expressions.append(referring_expression)

        return referring_expressions
    def generate_bfs_referring_expressions(self, n: int = 10) -> List[str]:
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

        for _ in range(n):
            template = random.choice(self.bfs_templates)

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

            # For "half" style templates that need two colors
            if any(phrase in template for phrase in ["half {color1} and half {color2}", "split between {color1} and {color2}",
                                                    "divided into {color1} and {color2}", "both {color1} and {color2}",
                                                    "half-and-half {color1} and {color2}"]):
                color1, color2 = random.choice(color_pairs)
                referring_expression = template.format(
                    shape_type=shape_type,
                    size=size,
                    color1=color1,
                    color2=color2,
                    style_desc=style_desc
                )
            # For "border" style templates
            elif any(phrase in template for phrase in ["outline", "border", "outlined", "framed", "edge"]):
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

        return referring_expressions
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
    print(f"- All attribute combinations: {len(all_combinations)}")
if __name__ == "__main__":
    main()
