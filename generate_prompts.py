#!/usr/bin/env python
"""
Generate templates for DFS (position-based) and BFS (attribute-based) prompts for the reason-synth dataset.
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

# Ordinal number mapping for natural language
ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"
}

class PromptGenerator:
    def __init__(self, max_grid_size: Tuple[int, int] = (8, 7)):
        """
        Initialize the prompt generator with the maximum grid size.
        Args:
            max_grid_size: Maximum grid size (rows, columns) in the dataset
        """
        self.max_rows, self.max_cols = max_grid_size
        self.dfs_templates = []
        self.bfs_templates = []
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all templates for both DFS and BFS prompts."""
        # DFS (position-based) templates
        self.dfs_templates = [
            "What is the {row_num} row {col_num} object?",
            "Describe the object at position ({row}, {col}).",
            "What is the {ordinal_col} object in the {ordinal_row} row?",
            "What color is the object in the {row_num} row, {col_num} column?",
            "What is the shape type of the object at position ({row}, {col})?",
            "What size is the {ordinal_col} object in the {ordinal_row} row?",
            "What style does the object at row {row}, column {col} have?",
            "Is the shape at position ({row}, {col}) a circle, triangle, or square?",
            "How many objects are there before the object at row {row}, column {col}?",
            "What is the object at row {row_num}, column {col_num} made of?"
        ]

        # BFS (attribute-based) templates
        self.bfs_templates = [
            # Single attribute templates
            "Find the {shape_type} in the image.",
            "Where is the {color} object?",
            "Locate the {size} shape.",
            "Which objects have a {style} style?",
            
            # Two attribute combinations
            "Find the {color} {shape_type}.",
            "Where is the {size} {shape_type}?",
            "Locate the {style} {shape_type}.",
            "Find the {color} {size} object.",
            "Where is the {style} {color} shape?",
            "Locate the {size} object with {style} style.",
            
            # Three attribute combinations
            "Find the {size} {color} {shape_type}.",
            "Where is the {style} {color} {shape_type}?",
            "Locate the {size} {shape_type} with {style} style.",
            
            # Four attribute combination (all attributes)
            "Find the {size} {color} {shape_type} with {style} style.",
            
            # Special templates for "half" style (two colors)
            "Find the half {color1} half {color2} object.",
            "Where is the {shape_type} that is half {color1} and half {color2}?",
            "Locate the {size} shape that is half {color1} and half {color2}.",
            "Find the {size} {shape_type} that is half {color1} and half {color2}.",
            
            # Special templates for "border" style
            "Find the object with a {color} border.",
            "Where is the {shape_type} with a {color} border?",
            "Locate the {size} {shape_type} with a {color} border.",
            "Find the {color1} {shape_type} with a {color2} border.",
            
            # Counting and set operations
            "How many {color} objects are there?",
            "Count the {size} {shape_type}s.",
            "How many objects are {style}?",
            "Count the {shape_type}s that are either {color1} or {color2}.",
            "How many {shape_type}s have a {color} component?"
        ]

    def generate_dfs_prompts(self, n: int = 10) -> List[str]:
        """
        Generate a specified number of DFS (position-based) prompts.
        Args:
            n: Number of prompts to generate
        Returns:
            List of generated prompts
        """
        prompts = []
        for _ in range(n):
            template = random.choice(self.dfs_templates)
            row = random.randint(1, self.max_rows)
            col = random.randint(1, self.max_cols)
            
            # Get ordinals if needed
            ordinal_row = ORDINALS.get(row, f"{row}th")
            ordinal_col = ORDINALS.get(col, f"{col}th")
            
            # Format the template
            prompt = template.format(
                row=row, col=col,
                row_num=f"{row}{'st' if row == 1 else 'nd' if row == 2 else 'rd' if row == 3 else 'th'}",
                col_num=f"{col}{'st' if col == 1 else 'nd' if col == 2 else 'rd' if col == 3 else 'th'}",
                ordinal_row=ordinal_row,
                ordinal_col=ordinal_col
            )
            prompts.append(prompt)
        
        return prompts

    def generate_bfs_prompts(self, n: int = 10) -> List[str]:
        """
        Generate a specified number of BFS (attribute-based) prompts.
        Args:
            n: Number of prompts to generate
        Returns:
            List of generated prompts
        """
        prompts = []
        
        # Create all combinations of attributes for more complex prompts
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
            
            # For "half" style templates that need two colors
            if "half {color1} half {color2}" in template or "half {color1} and half {color2}" in template:
                color1, color2 = random.choice(color_pairs)
                prompt = template.format(
                    shape_type=shape_type,
                    size=size,
                    color1=color1,
                    color2=color2
                )
            # For "border" style templates
            elif "border" in template:
                color1 = random.choice(COLORS)
                color2 = random.choice([c for c in COLORS if c != color1])
                prompt = template.format(
                    shape_type=shape_type,
                    size=size,
                    color=color,
                    color1=color1,
                    color2=color2
                )
            # For other templates
            else:
                prompt = template.format(
                    shape_type=shape_type,
                    color=color,
                    size=size,
                    style=style,
                    color1=random.choice(COLORS),
                    color2=random.choice([c for c in COLORS if c != color])
                )
            
            prompts.append(prompt)
        
        return prompts

    def generate_all_attribute_combinations(self) -> List[str]:
        """
        Generate prompts covering all possible attribute combinations.
        Returns:
            List of prompts with all attribute combinations
        """
        prompts = []
        
        # Generate prompts for all shape types
        for shape in SHAPE_TYPES:
            prompts.append(f"Find all the {shape}s in the image.")
        
        # Generate prompts for all colors
        for color in COLORS:
            prompts.append(f"Find all {color} objects in the image.")
        
        # Generate prompts for all sizes
        for size in SIZES:
            prompts.append(f"Find all {size} objects in the image.")
        
        # Generate prompts for all styles
        for style in STYLES:
            prompts.append(f"Find all objects with {style} style in the image.")
        
        # Generate prompts for all shape-color combinations
        for shape, color in itertools.product(SHAPE_TYPES, COLORS):
            prompts.append(f"Find the {color} {shape}.")
        
        # Generate prompts for all size-shape combinations
        for size, shape in itertools.product(SIZES, SHAPE_TYPES):
            prompts.append(f"Find the {size} {shape}.")
        
        # Generate prompts for all style-shape combinations
        for style, shape in itertools.product(STYLES, SHAPE_TYPES):
            prompts.append(f"Find the {shape} with {style} style.")
        
        # Generate special prompts for half style with all color combinations
        for color1, color2 in itertools.combinations(COLORS, 2):
            prompts.append(f"Find the object that is half {color1} and half {color2}.")
            
            # Add shape-specific half-color prompts
            for shape in SHAPE_TYPES:
                prompts.append(f"Find the {shape} that is half {color1} and half {color2}.")
        
        # Generate special prompts for border style with all color combinations
        for main_color, border_color in itertools.product(COLORS, COLORS):
            if main_color != border_color:
                prompts.append(f"Find the {main_color} object with a {border_color} border.")
                
                # Add shape-specific border prompts
                for shape in SHAPE_TYPES:
                    prompts.append(f"Find the {main_color} {shape} with a {border_color} border.")
        
        return prompts

def main():
    """Generate and print prompt templates."""
    generator = PromptGenerator()
    
    print("=== DFS (Position-Based) Prompts ===")
    dfs_prompts = generator.generate_dfs_prompts(10)
    for i, prompt in enumerate(dfs_prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\n=== BFS (Attribute-Based) Prompts ===")
    bfs_prompts = generator.generate_bfs_prompts(10)
    for i, prompt in enumerate(bfs_prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\n=== All Attribute Combinations (Sample) ===")
    all_combinations = generator.generate_all_attribute_combinations()
    # Print just a sample of these combinations
    for i, prompt in enumerate(random.sample(all_combinations, min(10, len(all_combinations))), 1):
        print(f"{i}. {prompt}")
    
    print(f"\nTotal templates generated:")
    print(f"- DFS templates: {len(generator.dfs_templates)}")
    print(f"- BFS templates: {len(generator.bfs_templates)}")
    print(f"- All attribute combinations: {len(all_combinations)}")

if __name__ == "__main__":
    main()
