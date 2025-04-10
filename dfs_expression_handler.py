#!/usr/bin/env python
"""
Generate templates for DFS (position-based) referring expressions for the reason-synth dataset.

DFS (Depth-First Search) refers to position-based expressions where objects are identified by their grid position
(e.g., "the shape in the third row, second column")

This metaphor represents a systematic approach where we follow a clear path to identify an object,
like listing items one by one until we reach the target (position-based references are a specific case of this).

IMPORTANT REQUIREMENTS:
1. All referring expressions must use natural language that sounds like a human would ask
2. Avoid technical terminology like coordinates (x,y)
3. Referring expressions should be diverse and cover various ways to ask about objects
4. Remove question phrases like "find" or "where is" - just keep the expression itself
"""

import json
import random
from typing import List, Dict, Tuple

# Load the object configuration
CONFIG_PATH = "src/object_config.json"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

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

class DFSExpressionHandler:
    def __init__(self, max_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize the DFS referring expression handler with the maximum grid size.
        Args:
            max_grid_size: Maximum grid size (rows, columns) in the dataset
        """
        self.max_rows, self.max_cols = max_grid_size
        self.dfs_templates = []
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all templates for DFS referring expressions."""
        # DFS (position-based) templates as referring expressions
        self.dfs_templates = [
            # Core position templates with balanced direction references
            "the object in the {ordinal_row} row {row_dir}, {ordinal_col} column {col_dir}",
            "the object in the {ordinal_row} row {row_alt_dir}, {ordinal_col} column {col_dir}",
            "the object in the {ordinal_row} row {row_dir}, {ordinal_col} column {col_alt_dir}",
            "the object in the {ordinal_row} row {row_alt_dir}, {ordinal_col} column {col_alt_dir}",

            # Variations on position references
            "the {ordinal_col} object {col_dir} in the {ordinal_row} row {row_dir}",
            "the {ordinal_col} object {col_alt_dir} in the {ordinal_row} row {row_dir}",
            "the {ordinal_col} object {col_dir} in the {ordinal_row} row {row_alt_dir}",
            "the {ordinal_col} object {col_alt_dir} in the {ordinal_row} row {row_alt_dir}",

            # Alternative phrasing
            "the shape in row {row_num} {row_dir}, column {col_num} {col_dir}",
            "the shape in row {row_num} {row_alt_dir}, column {col_num} {col_dir}",
            "the shape in row {row_num} {row_dir}, column {col_num} {col_alt_dir}",
            "the shape in row {row_num} {row_alt_dir}, column {col_num} {col_alt_dir}",

            # Directional counting
            "starting from the {row_dir}, the object in row {ordinal_row}, column {ordinal_col} {col_dir}",
            "starting from the {row_alt_dir}, the object in row {ordinal_row}, column {ordinal_col} {col_alt_dir}",
            "starting from the {col_dir}, the {ordinal_col} object in the {ordinal_row} row {row_alt_dir}",
            "starting from the {col_alt_dir}, the {ordinal_col} object in the {ordinal_row} row {row_dir}",
        ]

    def generate_dfs_referring_expressions(self, n: int = 10, min_grid: Tuple[int, int] = None, max_grid: Tuple[int, int] = None, reasoning_level: int = None) -> List[Dict]:
        """
        Generate a specified number of DFS (position-based) referring expressions.
        Args:
            n: Number of referring expressions to generate
            min_grid: Minimum grid size (rows, columns) to consider, defaults to (0, 0)
            max_grid: Maximum grid size (rows, columns) to consider, defaults to self.max_grid_size
            reasoning_level: Specific reasoning level to generate expressions for, defaults to None (random)
        Returns:
            List of dictionaries containing:
                - referring_expression: the formatted referring expression text
                - requirements: dictionary of requirements (row and column position)
        """
        referring_expressions = []

        # Set default grid limits if not provided
        min_row, min_col = min_grid if min_grid else (0, 0)
        max_row, max_col = max_grid if max_grid else (self.max_rows - 1, self.max_cols - 1)

        expressions_generated = 0
        max_attempts = n * 10  # Limit attempts to avoid infinite loops
        attempts = 0

        while expressions_generated < n and attempts < max_attempts:
            attempts += 1
            template = random.choice(self.dfs_templates)

            # If reasoning_level is specified, we need to find row and column that sum to the target level
            if reasoning_level is not None:
                # The user wants reasoning_level to be (display_row + display_col) - 2
                # So, the target sum for display_row + display_col is reasoning_level + 2
                target_display_sum = reasoning_level + 2

                possible_rows = []
                for r in range(min_row, max_row + 1):
                    display_r = r + 1
                    target_display_col = target_display_sum - display_r
                    col = target_display_col - 1  # Convert to 0-indexed

                    if min_col <= col <= max_col:
                        possible_rows.append(r)

                if not possible_rows:
                    # If no combinations found for the requested level, skip or raise error?
                    # For now, let's skip and hope we find enough in other attempts.
                    # Consider adding a check after the loop if expressions_generated < n.
                    continue

                # Randomly select a valid row
                row = random.choice(possible_rows)
                # Calculate the corresponding column based on the target sum
                display_row = row + 1
                display_col = target_display_sum - display_row
                col = display_col - 1  # Convert to 0-indexed
            else:
                # Random selection for row and column
                row = random.randint(min_row, max_row)  # Using 0-indexed grid positions
                col = random.randint(min_col, max_col)  # Using 0-indexed grid positions
                display_row = row + 1
                display_col = col + 1

            # Calculate the final reasoning level based on the new definition
            current_reasoning_level = (display_row + display_col) - 2

            # Get ordinals (adding 1 for human-readable numbers)
            ordinal_row = ORDINALS.get(display_row, f"{display_row}th")
            ordinal_col = ORDINALS.get(display_col, f"{display_col}th")

            # Select direction specifications based on template requirements
            row_dir = row_alt_dir = col_dir = col_alt_dir = "not defined"
            row_dir_type = "not_applicable" # Default if not set
            col_dir_type = "not_applicable" # Default if not set
            if "{row_dir}" in template:
                row_dir = random.choice(DIRECTION_SPECS["row_from_top"])
                row_dir_type = "row_from_top"
            if "{row_alt_dir}" in template:
                row_alt_dir = random.choice(DIRECTION_SPECS["row_from_bottom"])
                row_dir_type = "row_from_bottom"
            if "{col_dir}" in template:
                col_dir = random.choice(DIRECTION_SPECS["col_from_left"])
                col_dir_type = "col_from_left"
            if "{col_alt_dir}" in template:
                col_alt_dir = random.choice(DIRECTION_SPECS["col_from_right"])
                col_dir_type = "col_from_right"

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
                    "column": col,  # 0-indexed position for internal use
                    "row_dir_type": row_dir_type,  # Direction type used for rows
                    "col_dir_type": col_dir_type,  # Direction type used for columns
                    "display_row": display_row,  # 1-indexed position for display
                    "display_col": display_col,  # 1-indexed position for display
                },
                # Use the newly defined reasoning level
                "reasoning_level": current_reasoning_level
            }
            referring_expressions.append(expr_data)
            expressions_generated += 1

        # Check if enough expressions were generated, especially if a reasoning level was specified
        if expressions_generated < n and reasoning_level is not None:
            print(f"Warning: Only generated {expressions_generated}/{n} expressions for reasoning level {reasoning_level}. "
                  f"Possible that no valid row/column combinations exist within the grid limits.")

        return referring_expressions

    def find_matching_objects(self, requirements: Dict, objects: List[Dict], grid_size: Tuple[int, int]) -> List[Dict]:
        """
        Find all objects that match the given DFS requirements.

        Args:
            requirements: Dictionary of requirements from the referring expression
            objects: List of objects to check
            grid_size: Size of the grid (rows, columns)

        Returns:
            List of matching objects
        """
        matching_objects = []

        # Extract grid dimensions
        num_rows, num_cols = grid_size

        # Check if direction information is available in requirements
        row_dir_type = requirements.get('row_dir_type', 'row_from_top')
        col_dir_type = requirements.get('col_dir_type', 'col_from_left')

        # Extract target positions
        target_row = requirements['row']  # 0-indexed position
        target_col = requirements['column']  # 0-indexed position

        for obj in objects:
            # Get the object's grid position
            obj_row, obj_col = obj['grid_position']

            # Convert object position based on direction for comparison
            # For "from bottom" direction, invert the row position
            if row_dir_type == 'row_from_bottom':
                effective_obj_row = num_rows - 1 - obj_row
            else:
                effective_obj_row = obj_row

            # For "from right" direction, invert the column position
            if col_dir_type == 'col_from_right':
                effective_obj_col = num_cols - 1 - obj_col
            else:
                effective_obj_col = obj_col

            # Check if the effective position matches the target position
            if target_row == effective_obj_row and target_col == effective_obj_col:
                matching_objects.append(obj)

        return matching_objects

    def generate_random_objects(self, count: int = 10, min_grid: Tuple[int, int] = None, max_grid: Tuple[int, int] = None) -> List[Dict]:
        """
        Generate a list of objects with random grid positions.

        Args:
            count: Number of random objects to generate
            min_grid: Minimum grid size (rows, columns) to consider, defaults to (0, 0)
            max_grid: Maximum grid size (rows, columns) to consider, defaults to self.max_grid_size

        Returns:
            List of objects with random grid positions
        """
        # Set default grid limits if not provided
        min_row, min_col = min_grid if min_grid else (0, 0)
        max_row, max_col = max_grid if max_grid else (self.max_rows - 1, self.max_cols - 1)

        random_objects = []

        for _ in range(count):
            # Generate a random object with random grid position
            row = random.randint(min_row, max_row)
            col = random.randint(min_col, max_col)

            random_obj = {
                "grid_position": [row, col],
                # Add dummy shape attributes to make object structure consistent
                "shape_type": "random",
                "color1": "random",
                "color2": "random",
                "size": "random",
                "style": "random"
            }

            random_objects.append(random_obj)

        return random_objects

if __name__ == "__main__":
    """Generate and print DFS referring expression templates."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate DFS referring expression templates")
    parser.add_argument("--count", type=int, default=10, help="Number of DFS expressions to generate")
    parser.add_argument("--min_grid_row", type=int, default=0, help="Minimum grid row (0-indexed)")
    parser.add_argument("--min_grid_col", type=int, default=0, help="Minimum grid column (0-indexed)")
    parser.add_argument("--max_grid_row", type=int, default=7, help="Maximum grid row (0-indexed)")
    parser.add_argument("--max_grid_col", type=int, default=7, help="Maximum grid column (0-indexed)")
    parser.add_argument("--reasoning_level", "-r", type=int, default=None, help="Specific reasoning level to target (sum of row and column positions)")

    args = parser.parse_args()

    # Initialize generator with maximum grid size
    handler = DFSExpressionHandler(max_grid_size=(8, 8))

    print("=== DFS (Position-Based) Referring Expressions ===")
    print(f"Grid range: ({args.min_grid_row},{args.min_grid_col}) to ({args.max_grid_row},{args.max_grid_col})")
    if args.reasoning_level:
        print(f"Targeting reasoning level: {args.reasoning_level}")

    dfs_referring_expressions = handler.generate_dfs_referring_expressions(
        args.count,
        min_grid=(args.min_grid_row, args.min_grid_col),
        max_grid=(args.max_grid_row, args.max_grid_col),
        reasoning_level=args.reasoning_level
    )

    for i, expr_data in enumerate(dfs_referring_expressions, 1):
        print(f"{i}. Expression: {expr_data['referring_expression']}")
        print(f"   Requirements: row={expr_data['target_requirements']['row']}, column={expr_data['target_requirements']['column']}")
        print(f"   Direction types: row={expr_data['target_requirements']['row_dir_type']}, column={expr_data['target_requirements']['col_dir_type']}")
        print(f"   Reasoning Level: {expr_data['reasoning_level']}")
        print()

    print(f"\nTotal templates: {len(handler.dfs_templates)}")