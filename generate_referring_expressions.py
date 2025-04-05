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
import argparse

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
    def __init__(self, max_grid_size: Tuple[int, int] = (8, 8)):
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

    def generate_dfs_referring_expressions(self, n: int = 10, min_grid: Tuple[int, int] = None, max_grid: Tuple[int, int] = None) -> List[Dict]:
        """
        Generate a specified number of DFS (position-based) referring expressions.
        Args:
            n: Number of referring expressions to generate
            min_grid: Minimum grid size (rows, columns) to consider, defaults to (0, 0)
            max_grid: Maximum grid size (rows, columns) to consider, defaults to self.max_grid_size
        Returns:
            List of dictionaries containing:
                - referring_expression: the formatted referring expression text
                - requirements: dictionary of requirements (row and column position)
        """
        referring_expressions = []

        # Set default grid limits if not provided
        min_row, min_col = min_grid if min_grid else (0, 0)
        max_row, max_col = max_grid if max_grid else (self.max_rows - 1, self.max_cols - 1)

        for _ in range(n):
            template = random.choice(self.dfs_templates)
            row = random.randint(min_row, max_row)  # Using 0-indexed grid positions
            col = random.randint(min_col, max_col)  # Using 0-indexed grid positions

            # Get ordinals (adding 1 for human-readable numbers)
            display_row = row + 1
            display_col = col + 1
            ordinal_row = ORDINALS.get(display_row, f"{display_row}th")
            ordinal_col = ORDINALS.get(display_col, f"{display_col}th")

            # Select random direction specifications
            row_dir = random.choice(DIRECTION_SPECS["row_from_top"])
            row_alt_dir = random.choice(DIRECTION_SPECS["row_from_bottom"])
            col_dir = random.choice(DIRECTION_SPECS["col_from_left"])
            col_alt_dir = random.choice(DIRECTION_SPECS["col_from_right"])

            # Format the template
            referring_expression = template.format(
                row=display_row, col=display_col,
                row_num=f"{display_row}{'st' if display_row == 1 else 'nd' if display_row == 2 else 'rd' if display_row == 3 else 'th'}",
                col_num=f"{display_col}{'st' if display_col == 1 else 'nd' if display_col == 2 else 'rd' if display_col == 3 else 'th'}",
                ordinal_row=ordinal_row,
                ordinal_col=ordinal_col,
                row_dir=row_dir,
                row_alt_dir=row_alt_dir,
                col_dir=col_dir,
                col_alt_dir=col_alt_dir
            )

            # Create the result including both expression and requirements
            expr_data = {
                "referring_expression": referring_expression,
                "expression_type": "DFS",
                "target_requirements": {
                    "row": row,  # 0-indexed position for internal use
                    "column": col  # 0-indexed position for internal use
                }
            }
            referring_expressions.append(expr_data)

        return referring_expressions

    def generate_bfs_referring_expressions(self, n: int = 10,
                               comprehensive_ratio: float = 0.15,
                               specific_requirements: Dict = None,
                               single_attr_ratio: float = 0.25,
                               two_attr_ratio: float = 0.25,
                               three_attr_ratio: float = 0.25,
                               all_attr_ratio: float = 0.25,
                               sampling_strategy: str = "mixed",
                               objects: List[Dict] = None) -> List[Dict]:
        """
        Generate a specified number of BFS (attribute-based) referring expressions.
        Args:
            n: Number of referring expressions to generate
            comprehensive_ratio: Ratio of expressions to generate from comprehensive combinations
            specific_requirements: Dictionary of specific requirements to enforce on generated expressions.
                                  Format: {attribute: value} or None for any value
                                  Supported attributes: shape_type, size, style, color1, color2
            single_attr_ratio: Proportion of single attribute expressions
            two_attr_ratio: Proportion of two attribute combinations
            three_attr_ratio: Proportion of three attribute combinations
            all_attr_ratio: Proportion of all attribute expressions
            sampling_strategy: Strategy for generating expressions ("existence", "random", or "mixed")
            objects: List of objects (required for existence-based sampling)
        Returns:
            List of dictionaries containing:
                - referring_expression: the formatted referring expression text
                - requirements: dictionary of requirements (attributes like shape, color, size, style)
        """
        referring_expressions = []

        # Default specific requirements to empty dict if None
        specific_requirements = specific_requirements or {}

        # If existence-based sampling is used but no objects provided, fall back to random sampling
        if sampling_strategy in ["existence", "mixed"] and objects is None:
            print("Warning: Existence-based sampling requires objects, falling back to random sampling.")
            sampling_strategy = "random"

        # Group pattern keys by attribute count
        single_attr_patterns = ["shape_only", "color_only", "size_only", "style_only"]
        two_attr_patterns = ["shape_color", "shape_size", "shape_style", "color_size", "color_style", "size_style"]
        three_attr_patterns = ["shape_color_size", "shape_color_style", "shape_size_style", "color_size_style"]
        all_attr_patterns = ["all_attributes"]

        # Create all combinations of attributes for more complex referring expressions
        shape_color_combos = list(itertools.product(SHAPE_TYPES, COLORS))
        size_shape_combos = list(itertools.product(SIZES, SHAPE_TYPES))
        color_pairs = list(itertools.combinations(COLORS, 2))

        # Determine how many expressions to generate with each strategy
        if sampling_strategy == "existence":
            existence_count = n
            random_count = 0
        elif sampling_strategy == "random":
            existence_count = 0
            random_count = n
        else:  # mixed
            existence_count = n // 2
            random_count = n - existence_count

        # Determine counts for each attribute category
        total_count = n
        single_attr_count = max(1, int(total_count * single_attr_ratio))
        two_attr_count = max(1, int(total_count * two_attr_ratio))
        three_attr_count = max(1, int(total_count * three_attr_ratio))
        all_attr_count = max(1, int(total_count * all_attr_ratio))

        # Adjust counts to ensure they sum to n
        total_allocated = single_attr_count + two_attr_count + three_attr_count + all_attr_count
        if total_allocated != total_count:
            # Distribute the difference proportionally
            diff = total_count - total_allocated
            if diff > 0:
                # Add extra expressions proportionally
                ratios = [single_attr_ratio, two_attr_ratio, three_attr_ratio, all_attr_ratio]
                counts = [single_attr_count, two_attr_count, three_attr_count, all_attr_count]
                for _ in range(diff):
                    idx = random.choices(range(4), weights=ratios)[0]
                    counts[idx] += 1
                single_attr_count, two_attr_count, three_attr_count, all_attr_count = counts
            else:
                # Remove expressions proportionally
                for _ in range(-diff):
                    if single_attr_count > 1:
                        single_attr_count -= 1
                    elif two_attr_count > 1:
                        two_attr_count -= 1
                    elif three_attr_count > 1:
                        three_attr_count -= 1
                    elif all_attr_count > 1:
                        all_attr_count -= 1

        # Process the EXISTENCE-BASED sampling
        if existence_count > 0 and objects:
            # Balance counts between existence and random sampling
            existence_single = max(1, int(single_attr_count * (existence_count / n)))
            existence_two = max(1, int(two_attr_count * (existence_count / n)))
            existence_three = max(1, int(three_attr_count * (existence_count / n)))
            existence_all = max(1, int(all_attr_count * (existence_count / n)))

            # Ensure we have at least one object
            if not objects:
                print("Warning: No objects provided for existence-based sampling.")
            else:
                # Generate expressions based on actual objects
                for _ in range(existence_count):
                    # Select a random object
                    obj = random.choice(objects)

                    # Determine which category to generate (single, two, three, all)
                    # based on how many of each we still need
                    remaining_categories = []
                    if existence_single > 0:
                        remaining_categories.append(("single", existence_single))
                    if existence_two > 0:
                        remaining_categories.append(("two", existence_two))
                    if existence_three > 0:
                        remaining_categories.append(("three", existence_three))
                    if existence_all > 0:
                        remaining_categories.append(("all", existence_all))

                    if not remaining_categories:
                        break

                    # Choose a category weighted by how many we still need
                    chosen_category, count = random.choices(
                        remaining_categories,
                        weights=[count for _, count in remaining_categories]
                    )[0]

                    # Decrement the count for this category
                    if chosen_category == "single":
                        existence_single -= 1
                        # Choose one attribute from the object
                        attr_options = ["shape", "color", "size", "style"]
                        chosen_attr = random.choice(attr_options)

                        requirements = {}
                        if chosen_attr == "shape":
                            requirements["shape_type"] = obj["shape_type"]
                            pattern_key = "shape_only"
                        elif chosen_attr == "color":
                            requirements["color1"] = obj["color1"]
                            pattern_key = "color_only"
                        elif chosen_attr == "size":
                            requirements["size"] = obj["size"]
                            pattern_key = "size_only"
                        elif chosen_attr == "style":
                            requirements["style"] = obj["style"]
                            pattern_key = "style_only"

                    elif chosen_category == "two":
                        existence_two -= 1
                        # Choose two attributes from the object
                        attr_pairs = [
                            ("shape", "color"), ("shape", "size"), ("shape", "style"),
                            ("color", "size"), ("color", "style"), ("size", "style")
                        ]
                        chosen_pair = random.choice(attr_pairs)

                        requirements = {}
                        if "shape" in chosen_pair:
                            requirements["shape_type"] = obj["shape_type"]
                        if "color" in chosen_pair:
                            requirements["color1"] = obj["color1"]
                        if "size" in chosen_pair:
                            requirements["size"] = obj["size"]
                        if "style" in chosen_pair:
                            requirements["style"] = obj["style"]

                        if chosen_pair == ("shape", "color"):
                            pattern_key = "shape_color"
                        elif chosen_pair == ("shape", "size"):
                            pattern_key = "shape_size"
                        elif chosen_pair == ("shape", "style"):
                            pattern_key = "shape_style"
                        elif chosen_pair == ("color", "size"):
                            pattern_key = "color_size"
                        elif chosen_pair == ("color", "style"):
                            pattern_key = "color_style"
                        elif chosen_pair == ("size", "style"):
                            pattern_key = "size_style"

                    elif chosen_category == "three":
                        existence_three -= 1
                        # Choose three attributes from the object
                        attr_triples = [
                            ("shape", "color", "size"),
                            ("shape", "color", "style"),
                            ("shape", "size", "style"),
                            ("color", "size", "style")
                        ]
                        chosen_triple = random.choice(attr_triples)

                        requirements = {}
                        if "shape" in chosen_triple:
                            requirements["shape_type"] = obj["shape_type"]
                        if "color" in chosen_triple:
                            requirements["color1"] = obj["color1"]
                        if "size" in chosen_triple:
                            requirements["size"] = obj["size"]
                        if "style" in chosen_triple:
                            requirements["style"] = obj["style"]

                        if chosen_triple == ("shape", "color", "size"):
                            pattern_key = "shape_color_size"
                        elif chosen_triple == ("shape", "color", "style"):
                            pattern_key = "shape_color_style"
                        elif chosen_triple == ("shape", "size", "style"):
                            pattern_key = "shape_size_style"
                        elif chosen_triple == ("color", "size", "style"):
                            pattern_key = "color_size_style"

                    elif chosen_category == "all":
                        existence_all -= 1
                        # Use all attributes from the object
                        requirements = {
                            "shape_type": obj["shape_type"],
                            "color1": obj["color1"],
                            "size": obj["size"],
                            "style": obj["style"]
                        }
                        if obj["style"] in ["half", "border"] and "color2" in obj:
                            requirements["color2"] = obj["color2"]

                        pattern_key = "all_attributes"

                    # Select a template for this pattern
                    if pattern_key in self.bfs_template_patterns:
                        templates = self.bfs_template_patterns[pattern_key]
                        if templates:
                            template = random.choice(templates)

                            # Get style description
                            style = obj["style"]
                            if style in STYLE_DESCRIPTIONS:
                                style_desc = random.choice(STYLE_DESCRIPTIONS[style])
                            else:
                                style_desc = style

                            # Format the template based on requirements
                            shape_type = requirements.get("shape_type", "")
                            color = requirements.get("color1", "")
                            size = requirements.get("size", "")
                            color1 = requirements.get("color1", "")
                            color2 = requirements.get("color2", color1)

                            # Format the template
                            referring_expression = template.format(
                                shape_type=shape_type,
                                color=color,
                                size=size,
                                style_desc=style_desc.format(color=color) if "{color}" in style_desc else style_desc,
                                color1=color1,
                                color2=color2
                            )

                            # Add the expression to the list
                            expr_data = {
                                "referring_expression": referring_expression,
                                "expression_type": "BFS",
                                "target_requirements": requirements
                            }
                            referring_expressions.append(expr_data)

        # Process the RANDOM-BASED sampling
        if random_count > 0:
            # Balance counts between existence and random sampling
            random_single = single_attr_count - (existence_single if 'existence_single' in locals() else 0)
            random_two = two_attr_count - (existence_two if 'existence_two' in locals() else 0)
            random_three = three_attr_count - (existence_three if 'existence_three' in locals() else 0)
            random_all = all_attr_count - (existence_all if 'existence_all' in locals() else 0)

            # Distribute expressions across pattern categories
            patterns_per_category = {
                "single": random_single,
                "two": random_two,
                "three": random_three,
                "all": random_all
            }

            # Generate expressions for each category
            for category, count in patterns_per_category.items():
                if count <= 0:
                    continue

                if category == "single":
                    pattern_keys = single_attr_patterns
                elif category == "two":
                    pattern_keys = two_attr_patterns
                elif category == "three":
                    pattern_keys = three_attr_patterns
                else:  # all
                    pattern_keys = all_attr_patterns

                # Distribute within each category
                patterns_count = {}
                remaining = count

            # First pass: ensure at least one of each pattern if possible
            for key in pattern_keys:
                if remaining > 0:
                        patterns_count[key] = 1
                    remaining -= 1
                else:
                        patterns_count[key] = 0

                # Second pass: distribute remaining expressions
            if remaining > 0:
                for _ in range(remaining):
                    key = random.choice(pattern_keys)
                        patterns_count[key] += 1

            # Generate expressions based on the distribution
                for key, count in patterns_count.items():
                    if count == 0 or key not in self.bfs_template_patterns:
                    continue

                templates = self.bfs_template_patterns[key]

                for _ in range(count):
                    template = random.choice(templates)

                        # Select random attributes (or use specific requirements if provided)
                        shape_type = specific_requirements.get('shape_type') or random.choice(SHAPE_TYPES)
                        color = specific_requirements.get('color1') or random.choice(COLORS)
                        size = specific_requirements.get('size') or random.choice(SIZES)
                        style = specific_requirements.get('style') or random.choice(STYLES)

                    # Initialize requirements dictionary
                    requirements = {}

                    # Add basic requirements based on pattern type
                    if "shape" in key:
                        requirements["shape_type"] = shape_type
                        if "color" in key and key != "color_style":  # Special case for color_style to avoid confusion
                        requirements["color1"] = color
                    if "size" in key:
                        requirements["size"] = size
                    if "style" in key:
                        requirements["style"] = style

                    # Get a natural description for the selected style
                    if style in STYLE_DESCRIPTIONS:
                        style_desc = random.choice(STYLE_DESCRIPTIONS[style])
                    else:
                        style_desc = style  # Fallback

                    # For "half_style" templates that need two colors
                    if key == "half_style":
                            color1 = specific_requirements.get('color1') or random.choice(COLORS)
                            # Ensure color2 is different from color1 if not specified
                            if 'color2' in specific_requirements:
                                color2 = specific_requirements['color2']
                            else:
                                color2 = random.choice([c for c in COLORS if c != color1])

                        requirements["style"] = "half"
                        requirements["color1"] = color1
                        requirements["color2"] = color2

                        referring_expression = template.format(
                            shape_type=shape_type,
                            size=size,
                            color1=color1,
                            color2=color2,
                            style_desc=style_desc
                        )

                    # For "border_style" templates
                    elif key == "border_style":
                            color1 = specific_requirements.get('color1') or random.choice(COLORS)
                            if 'color2' in specific_requirements:
                                color2 = specific_requirements['color2']
                            else:
                        color2 = random.choice([c for c in COLORS if c != color1])

                        requirements["style"] = "border"

                        if "color1" in template:
                            requirements["color1"] = color1
                            requirements["color2"] = color2
                        else:
                            requirements["color1"] = color

                        referring_expression = template.format(
                            shape_type=shape_type,
                            size=size,
                            color=color,
                            color1=color1,
                            color2=color2,
                            style_desc=style_desc
                        )

                    # For "solid_style" templates
                    elif key == "solid_style":
                        requirements["style"] = "solid"
                        requirements["color1"] = color

                        referring_expression = template.format(
                            shape_type=shape_type,
                            size=size,
                            color=color,
                            style_desc=style_desc.format(color=color) if "{color}" in style_desc else style_desc
                        )

                    # For other templates (standard attribute templates)
                    else:
                        # Format the referring expression and update requirements as needed
                        formatted_style_desc = style_desc.format(color=color) if "{color}" in style_desc else style_desc

                        # If multiple colors mentioned, add both to requirements
                        if "{color1}" in template and "{color2}" in template:
                                color1 = specific_requirements.get('color1') or random.choice(COLORS)
                                if 'color2' in specific_requirements:
                                    color2 = specific_requirements['color2']
                                else:
                            color2 = random.choice([c for c in COLORS if c != color1])

                            requirements["color1"] = color1
                            requirements["color2"] = color2
                        elif "color" in key:
                            requirements["color1"] = color

                            # Format with available colors
                            c1 = color1 if "{color1}" in template else color
                            c2 = color2 if "{color2}" in template else random.choice([c for c in COLORS if c != color])

                        referring_expression = template.format(
                            shape_type=shape_type,
                            color=color,
                            size=size,
                            style_desc=formatted_style_desc,
                                color1=c1,
                                color2=c2
                        )

                    # Create the full result dictionary
                    expr_data = {
                        "referring_expression": referring_expression,
                        "expression_type": "BFS",
                        "target_requirements": requirements
                    }
                    referring_expressions.append(expr_data)

        # Add comprehensive expressions if needed (legacy support)
        num_comprehensive = max(0, n - len(referring_expressions))
        if num_comprehensive > 0:
            all_combinations = self.generate_all_attribute_combinations(include_requirements=True)

            # Filter combinations based on specific requirements
            if specific_requirements:
                filtered_combinations = []
                for combo in all_combinations:
                    req = combo["target_requirements"]
                    match = True
                    for key, value in specific_requirements.items():
                        if value is not None and key in req and req[key] != value:
                            match = False
                            break
                    if match:
                        filtered_combinations.append(combo)
                all_combinations = filtered_combinations

            if all_combinations:
                comprehensive_samples = random.sample(all_combinations, min(num_comprehensive, len(all_combinations)))
                referring_expressions.extend(comprehensive_samples)

        # Shuffle the results and return the requested number
        random.shuffle(referring_expressions)
        return referring_expressions[:n]

    def generate_all_attribute_combinations(self, include_requirements=False) -> List:
        """
        Generate referring expressions covering all possible attribute combinations.

        Args:
            include_requirements: Whether to include requirements in the output

        Returns:
            List of referring expressions or dictionaries with requirements
        """
        result = []

        # Generate comprehensive attribute combinations
        for shape, size, style in itertools.product(SHAPE_TYPES, SIZES, STYLES):
            # Handle solid style
            if style == "solid":
                for color in COLORS:
                    expressions = [
                        f"the {size} {color} solid {shape}",
                        f"the completely filled {size} {color} {shape}",
                        f"the uniform {color} {size} {shape}"
                    ]

                    for expr in expressions:
                        if include_requirements:
                            result.append({
                                "referring_expression": expr,
                                "expression_type": "BFS",
                                "target_requirements": {
                                    "shape_type": shape,
                                    "size": size,
                                    "style": style,
                                    "color1": color
                                }
                            })
                        else:
                            result.append(expr)

            # Handle half style
            elif style == "half":
                for color1, color2 in itertools.combinations(COLORS, 2):
                    expressions = [
                        f"the {size} {shape} that's half {color1} and half {color2}",
                        f"the two-toned {size} {shape} in {color1} and {color2}",
                        f"the {size} {shape} divided between {color1} and {color2}"
                    ]

                    for expr in expressions:
                        if include_requirements:
                            result.append({
                                "referring_expression": expr,
                                "expression_type": "BFS",
                                "target_requirements": {
                                    "shape_type": shape,
                                    "size": size,
                                    "style": style,
                                    "color1": color1,
                                    "color2": color2
                                }
                            })
                        else:
                            result.append(expr)

            # Handle border style
            elif style == "border":
                for main_color, border_color in itertools.product(COLORS, COLORS):
                    if main_color != border_color:
                        expressions = [
                            f"the {size} {main_color} {shape} with a {border_color} outline",
                            f"the {main_color} {size} {shape} bordered in {border_color}",
                            f"the {size} {shape} that's {main_color} with a {border_color} frame"
                        ]

                        for expr in expressions:
                            if include_requirements:
                                result.append({
                                    "referring_expression": expr,
                                    "expression_type": "BFS",
                                    "target_requirements": {
                                        "shape_type": shape,
                                        "size": size,
                                        "style": style,
                                        "color1": main_color,
                                        "color2": border_color
                                    }
                                })
                            else:
                                result.append(expr)

        # Add additional combinations with DFS patterns (for completeness in the dataset)
        for shape, color, size in itertools.product(SHAPE_TYPES, COLORS, SIZES):
            expressions = [
                f"the {color} {size} {shape}",
                f"the {size} {shape} in {color}",
                f"the {color} {shape} of {size} size"
            ]

            for expr in expressions:
                if include_requirements:
                    result.append({
                        "referring_expression": expr,
                        "expression_type": "BFS",
                        "target_requirements": {
                            "shape_type": shape,
                            "size": size,
                            "color1": color
                        }
                    })
                else:
                    result.append(expr)

        return result

def main():
    """Generate and print referring expression templates."""
    parser = argparse.ArgumentParser(description="Generate referring expression templates")
    parser.add_argument("--dfs_count", type=int, default=5, help="Number of DFS expressions to generate")
    parser.add_argument("--bfs_count", type=int, default=5, help="Number of BFS expressions to generate")
    parser.add_argument("--min_grid_row", type=int, default=0, help="Minimum grid row (0-indexed)")
    parser.add_argument("--min_grid_col", type=int, default=0, help="Minimum grid column (0-indexed)")
    parser.add_argument("--max_grid_row", type=int, default=7, help="Maximum grid row (0-indexed)")
    parser.add_argument("--max_grid_col", type=int, default=7, help="Maximum grid column (0-indexed)")
    parser.add_argument("--shape_type", type=str, default=None, help="Specific shape type requirement for BFS")
    parser.add_argument("--size", type=str, default=None, help="Specific size requirement for BFS")
    parser.add_argument("--style", type=str, default=None, help="Specific style requirement for BFS")
    parser.add_argument("--color1", type=str, default=None, help="Specific primary color requirement for BFS")
    parser.add_argument("--color2", type=str, default=None, help="Specific secondary color requirement for BFS")

    args = parser.parse_args()

    # Initialize generator with maximum grid size
    generator = ReferringExpressionGenerator(max_grid_size=(8, 8))

    # Create specific requirements dictionary for BFS if any are provided
    specific_requirements = {}
    if args.shape_type:
        specific_requirements['shape_type'] = args.shape_type
    if args.size:
        specific_requirements['size'] = args.size
    if args.style:
        specific_requirements['style'] = args.style
    if args.color1:
        specific_requirements['color1'] = args.color1
    if args.color2:
        specific_requirements['color2'] = args.color2

    specific_requirements = specific_requirements or None  # Set to None if empty

    print("=== DFS (Position-Based) Referring Expressions ===")
    print(f"Grid range: ({args.min_grid_row},{args.min_grid_col}) to ({args.max_grid_row},{args.max_grid_col})")

    dfs_referring_expressions = generator.generate_dfs_referring_expressions(
        args.dfs_count,
        min_grid=(args.min_grid_row, args.min_grid_col),
        max_grid=(args.max_grid_row, args.max_grid_col)
    )

    for i, expr_data in enumerate(dfs_referring_expressions, 1):
        print(f"{i}. Expression: {expr_data['referring_expression']}")
        print(f"   Requirements: row={expr_data['target_requirements']['row']}, column={expr_data['target_requirements']['column']}")
        print()

    print("\n=== BFS (Attribute-Based) Referring Expressions ===")
    if specific_requirements:
        print(f"Using specific requirements: {specific_requirements}")

    bfs_referring_expressions = generator.generate_bfs_referring_expressions(
        args.bfs_count,
        specific_requirements=specific_requirements
    )

    for i, expr_data in enumerate(bfs_referring_expressions, 1):
        print(f"{i}. Expression: {expr_data['referring_expression']}")
        print(f"   Requirements: {', '.join([f'{k}={v}' for k, v in expr_data['target_requirements'].items()])}")
        print()

    # Print some comprehensive combinations
    print("\n=== All Attribute Combinations (Sample) ===")
    all_combinations = generator.generate_all_attribute_combinations(include_requirements=True)
    # Print just a sample of these combinations
    for i, expr_data in enumerate(random.sample(all_combinations, min(5, len(all_combinations))), 1):
        print(f"{i}. Expression: {expr_data['referring_expression']}")
        print(f"   Requirements: {', '.join([f'{k}={v}' for k, v in expr_data['target_requirements'].items()])}")
        print()

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
