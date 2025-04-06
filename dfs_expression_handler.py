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

        for obj in objects:
            # Get the object's grid position
            row, col = obj['grid_position']

            # Check both row and column requirements are present (DFS requires both coordinates)
            if 'row' not in requirements or 'column' not in requirements:
                continue

            # Check if the position matches exactly
            if requirements['row'] == row and requirements['column'] == col:
                matching_objects.append(obj)

        return matching_objects

if __name__ == "__main__":
    """Generate and print DFS referring expression templates."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate DFS referring expression templates")
    parser.add_argument("--count", type=int, default=5, help="Number of DFS expressions to generate")
    parser.add_argument("--min_grid_row", type=int, default=0, help="Minimum grid row (0-indexed)")
    parser.add_argument("--min_grid_col", type=int, default=0, help="Minimum grid column (0-indexed)")
    parser.add_argument("--max_grid_row", type=int, default=7, help="Maximum grid row (0-indexed)")
    parser.add_argument("--max_grid_col", type=int, default=7, help="Maximum grid column (0-indexed)")

    args = parser.parse_args()

    # Initialize generator with maximum grid size
    handler = DFSExpressionHandler(max_grid_size=(8, 8))

    print("=== DFS (Position-Based) Referring Expressions ===")
    print(f"Grid range: ({args.min_grid_row},{args.min_grid_col}) to ({args.max_grid_row},{args.max_grid_col})")

    dfs_referring_expressions = handler.generate_dfs_referring_expressions(
        args.count,
        min_grid=(args.min_grid_row, args.min_grid_col),
        max_grid=(args.max_grid_row, args.max_grid_col)
    )

    for i, expr_data in enumerate(dfs_referring_expressions, 1):
        print(f"{i}. Expression: {expr_data['referring_expression']}")
        print(f"   Requirements: row={expr_data['target_requirements']['row']}, column={expr_data['target_requirements']['column']}")
        print()

    print(f"\nTotal templates: {len(handler.dfs_templates)}")