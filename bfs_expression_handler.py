"""
BFS Expression Handler module for Reason-Synth dataset.

This module provides a class that handles both the generation and matching
of BFS (attribute-based) referring expressions.
"""

import copy
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass

# Constants (importing from existing constants)
SHAPE_TYPES = ["circle", "triangle", "square"]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
SIZES = ["small", "big"]
STYLES = ["solid", "half", "border"]

@dataclass
class Requirements:
    """Class to hold object property requirements for template matching."""
    shape_type: Optional[str] = None
    color1: Optional[str] = None
    color2: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert requirements to a dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BFSExpressionHandler:
    """
    Handler for generating and matching BFS (Basic Feature Selection) referring expressions.

    This class handles:
    1. Generating BFS expressions with defined templates and attribute requirements
    2. Matching objects to BFS expression requirements

    Requirements Logic:
    - Requirements are pre-computed during initialization
    - Each template has an associated list of requirement objects
    - Special cases like "only containing {color1} and {color2}" have multiple requirement objects
    """

    def __init__(self):
        """Initialize the BFS Expression Handler with templates and matching functions."""
        self.initialize_templates()

    def extract_requirements_from_template(self, template: str, style: Optional[str] = None) -> List[Requirements]:
        """
        Extract requirements from a template string based on placeholders.

        Args:
            template: The template string
            style: The style to use for style-specific requirements

        Returns:
            List of Requirements objects
        """
        # Create a base requirement
        base_req = Requirements()

        # Set style if provided
        if style:
            base_req.style = style

        for placeholder in ["{shape_type}", "{size}", "{color1}", "{color2}", "{style}"]:
            if placeholder in template:
                setattr(base_req, placeholder.replace("{", "").replace("}", ""), placeholder)

        # Default case: just return the base requirement
        requirements = [base_req]

        # Special case: templates with "only containing {color1} and {color2}"
        if "containing {color1} and {color2}" in template:
            # Create requirement for just color2
            color2_req = copy.deepcopy(base_req)
            color2_req.color1 = "{color2}"  # color2 as color1 for matching
            color2_req.color2 = "{color1}"  # color1 as color2 for matching
            requirements.append(color2_req)
        elif "containing {color1}" in template:
            # Create requirement for just color1
            color1_req = copy.deepcopy(base_req)
            color1_req.color2 = "{color1}"  # color1 as color1 for matching
            color1_req.color1 = None  # color1 as color2 for matching
            requirements.append(color1_req)
        elif "containing {color2}" in template:
            # Create requirement for just color2
            color2_req = copy.deepcopy(base_req)
            color2_req.color1 = "{color2}"  # color2 as color1 for matching
            color2_req.color2 = None  # color1 as color2 for matching
            requirements.append(color2_req)

        # Special case: half-style templates with both colors
        if style == "half" and "{color1}" in template and "{color2}" in template:
            # Create alternative with swapped colors
            alt_req = copy.deepcopy(base_req)
            alt_req.color1 = "{color2}"
            alt_req.color2 = "{color1}"

            requirements.append(alt_req)

        return requirements

    def template_with_requirements(self, template: str, style: Optional[str] = None) -> Tuple[str, List[Requirements]]:
        """
        Helper function to create a template tuple with pre-computed requirements.

        Args:
            template: The template string
            style: Optional style to use for style-specific requirements

        Returns:
            Tuple of (template_string, requirements_list)
        """
        return (template, self.extract_requirements_from_template(template, style))

    def initialize_templates(self):
        """Initialize templates for BFS referring expressions with pre-computed requirements."""
        # BFS templates organized by attribute patterns
        # For each pattern, we create style-specific subkeys for more natural language
        # Each template is now a tuple: (template_string, [requirements_list])

        self.template_patterns = {
            # Single attribute templates
            "shape": [
                self.template_with_requirements("the {shape_type}")
            ],

            # Color templates - templates that explicitly mention containing colors
            "color": [
                self.template_with_requirements("the object containing {color1}"),
                self.template_with_requirements("the object containing {color2}"),
                self.template_with_requirements("the object only containing {color1} and {color2}")
            ],

            # Size templates
            "size": [
                self.template_with_requirements("the {size} shape"),
                self.template_with_requirements("the {size} object")
            ],

            # Style templates with specific style variations
            "style": {
                "solid": [
                    self.template_with_requirements("the single-color object", "solid"),
                    self.template_with_requirements("the completely filled object", "solid")
                ],
                "half": [
                    self.template_with_requirements("the half and half object", "half"),
                    self.template_with_requirements("the equal-sized dual-colored object", "half")
                ],
                "border": [
                    self.template_with_requirements("the object with a border", "border"),
                    self.template_with_requirements("the outlined object", "border")
                ]
            },

            # Two attribute combinations - explicitly mentioning containing colors
            "shape_color": [
                self.template_with_requirements("the {shape_type} containing {color1}"),
                self.template_with_requirements("the {shape_type} containing {color2}"),
                self.template_with_requirements("the {shape_type} only containing {color1} and {color2}")
            ],

            "shape_size": [
                self.template_with_requirements("the {size} {shape_type}"),
                self.template_with_requirements("the {shape_type} that is {size}")
            ],

            "shape_style": {
                "solid": [
                    self.template_with_requirements("the single-color {shape_type}", "solid"),
                    self.template_with_requirements("the completely filled {shape_type}", "solid")
                ],
                "half": [
                    self.template_with_requirements("the half and half {shape_type}", "half"),
                    self.template_with_requirements("the {shape_type} split half-half into two colors", "half")
                ],
                "border": [
                    self.template_with_requirements("the {shape_type} with a border", "border"),
                    self.template_with_requirements("the outlined {shape_type}", "border")
                ]
            },

            "color_size": [
                self.template_with_requirements("the {size} object containing {color1}"),
                self.template_with_requirements("the {size} object containing {color2}"),
                self.template_with_requirements("the {size} object only containing {color1} and {color2}")
            ],

            "color_style": {
                "solid": [
                    self.template_with_requirements("the single-color {color1} object", "solid"),
                    self.template_with_requirements("the completely filled {color1} object", "solid")
                ],
                "half": [
                    self.template_with_requirements("the half {color1} half {color2} object", "half"),
                    self.template_with_requirements("the object split half-half into {color1} and {color2}", "half")
                ],
                "border": [
                    self.template_with_requirements("the {color1} object with a {color2} border", "border"),
                    self.template_with_requirements("the {color1} shape outlined in {color2}", "border")
                ]
            },

            "size_style": {
                "solid": [
                    self.template_with_requirements("the single-color {size} object", "solid"),
                    self.template_with_requirements("the completely filled {size} object", "solid")
                ],
                "half": [
                    self.template_with_requirements("the {size} object split half-half into two colors", "half"),
                    self.template_with_requirements("the equal-sized dual-colored {size} object", "half")
                ],
                "border": [
                    self.template_with_requirements("the {size} object with a border", "border"),
                    self.template_with_requirements("the outlined {size} object", "border")
                ]
            },

            # Three attribute combinations - explicitly mentioning containing colors
            "shape_color_size": [
                self.template_with_requirements("the {size} {shape_type} containing {color1}"),
                self.template_with_requirements("the {size} {shape_type} containing {color2}"),
                self.template_with_requirements("the {size} {shape_type} only containing {color1} and {color2}")
            ],

            "shape_color_style": {
                "solid": [
                    self.template_with_requirements("the single-color {color1} {shape_type}", "solid"),
                    self.template_with_requirements("the completely filled {color1} {shape_type}", "solid")
                ],
                "half": [
                    self.template_with_requirements("the half {color1} half {color2} {shape_type}", "half"),
                    self.template_with_requirements("the {shape_type} split half-half into {color1} and {color2}", "half")
                ],
                "border": [
                    self.template_with_requirements("the {color1} {shape_type} with a {color2} border", "border"),
                    self.template_with_requirements("the {shape_type} that's {color1} outlined in {color2}", "border")
                ]
            },

            "shape_size_style": {
                "solid": [
                    self.template_with_requirements("the single-color {size} {shape_type}", "solid"),
                    self.template_with_requirements("the {size} {shape_type} that's completely filled", "solid")
                ],
                "half": [
                    self.template_with_requirements("the {size} {shape_type} split half-half into two colors", "half"),
                    self.template_with_requirements("the equal-sized dual-colored {size} {shape_type}", "half")
                ],
                "border": [
                    self.template_with_requirements("the {size} {shape_type} with a border", "border"),
                    self.template_with_requirements("the outlined {size} {shape_type}", "border")
                ]
            },

            "color_size_style": {
                "solid": [
                    self.template_with_requirements("the single-color {color1} {size} object", "solid"),
                    self.template_with_requirements("the completely filled {color1} {size} object", "solid")
                ],
                "half": [
                    self.template_with_requirements("the {size} object split half-half into {color1} and {color2}", "half"),
                    self.template_with_requirements("the half {color1} half {color2} {size} object", "half")
                ],
                "border": [
                    self.template_with_requirements("the {size} {color1} object with a {color2} border", "border"),
                    self.template_with_requirements("the {color1} {size} object outlined in {color2}", "border")
                ]
            },

            "shape_color_size_style": {
                "solid": [
                    self.template_with_requirements("the single-color {color1} {size} {shape_type}", "solid"),
                    self.template_with_requirements("the completely filled {color1} {size} {shape_type}", "solid"),
                    self.template_with_requirements("the {size} {shape_type} that's purely {color1}", "solid")
                ],
                "half": [
                    self.template_with_requirements("the {size} {shape_type} split half-half into {color1} and {color2}", "half"),
                    self.template_with_requirements("the half {color1} half {color2} {size} {shape_type}", "half"),
                    self.template_with_requirements("the {size} {shape_type} that's half {color1}", "half"),
                    self.template_with_requirements("the {size} {shape_type} with half {color2}", "half")
                ],
                "border": [
                    self.template_with_requirements("the {size} {color1} {shape_type} with a {color2} border", "border"),
                    self.template_with_requirements("the {color1} {size} {shape_type} outlined in {color2}", "border"),
                    self.template_with_requirements("the {color1} {size} {shape_type} framed by {color2}", "border"),
                    self.template_with_requirements("the {size} {shape_type} that's mainly {color1}", "border"),
                    self.template_with_requirements("the {size} {shape_type} framed by {color2}", "border"),
                    self.template_with_requirements("the {size} {shape_type} outlined in {color2}", "border")
                ]
            },
        }

        # Create a flattened list of all templates for random selection
        self.all_templates = []
        for _, value in self.template_patterns.items():
            if isinstance(value, dict):
                # For style-specific templates, flatten all styles
                for style_templates in value.values():
                    self.all_templates.extend([template_tuple[0] for template_tuple in style_templates])
            else:
                # For standard templates
                self.all_templates.extend([template_tuple[0] for template_tuple in value])

        # Define the matching criteria for each template pattern
        self.pattern_matching_keys = {
            # Single attribute templates
            "shape": ["shape_type"],
            "color": ["color1", "color2"],  # Match either color for flexibility
            "size": ["size"],
            "style": ["style"],

            # Two attribute combinations
            "shape_color": ["shape_type", "color1", "color2"],  # Match shape and either color
            "shape_size": ["shape_type", "size"],
            "shape_style": ["shape_type", "style"],
            "color_size": ["size", "color1", "color2"],  # Match size and either color
            "color_style": ["style", "color1", "color2"],  # Match style and either color
            "size_style": ["size", "style"],

            # Three attribute combinations
            "shape_color_size": ["shape_type", "size", "color1", "color2"],
            "shape_color_style": ["shape_type", "style", "color1", "color2"],
            "shape_size_style": ["shape_type", "size", "style"],
            "color_size_style": ["size", "style", "color1", "color2"],

            # All attributes
            "shape_color_size_style": ["shape_type", "size", "style", "color1", "color2"],

            # Style-specific templates
            "solid_style": ["style", "color1", "shape_type"],
            "half_style": ["style", "color1", "color2", "shape_type"],
            "border_style": ["style", "color1", "color2", "shape_type"]
        }

        # Group templates by category
        self.single_attr_patterns = ["shape", "color", "size", "style"]
        self.two_attr_patterns = ["shape_color", "shape_size", "shape_style", "color_size", "color_style", "size_style"]
        self.three_attr_patterns = ["shape_color_size", "shape_color_style", "shape_size_style", "color_size_style"]
        self.all_attr_patterns = ["shape_color_size_style"]
        self.style_specific_patterns = ["solid_style", "half_style", "border_style"]

        # Create a pattern groups dictionary for ratio-based sampling
        self.pattern_groups = {
            "single": self.single_attr_patterns,
            "two": self.two_attr_patterns,
            "three": self.three_attr_patterns,
            "all": self.all_attr_patterns
        }

    def matches_attribute_requirements(self, obj_info, requirement):
        """
        Check if an object matches a single attribute requirement.

        Args:
            obj_info: Dictionary of object attributes
            requirement: Dictionary or Requirements object of a single required attribute set

        Returns:
            Boolean indicating if object matches requirements
        """
        # Convert Requirements object to dict if needed
        if isinstance(requirement, Requirements):
            requirement = requirement.to_dict()

        for attr, required_value in requirement.items():
            if attr not in obj_info:
                return False

            obj_value = obj_info[attr]

            # Skip placeholder values (used in template definition)
            if required_value in ["{shape_type}", "{color1}", "{color2}", "{size}", "{style}"]:
                continue

            if obj_value != required_value:
                return False

        return True

    def matches_any_requirements(self, obj_info, requirements_list):
        """
        Check if an object matches any of the requirements in the list.

        Args:
            obj_info: Dictionary of object attributes
            requirements_list: List of requirement dictionaries or Requirements objects

        Returns:
            Boolean indicating if object matches any requirement
        """
        for requirement in requirements_list:
            if self.matches_attribute_requirements(obj_info, requirement):
                return True
        return False

    def find_matching_objects(self, requirements_list, objects):
        """
        Find all objects that match any of the given BFS requirements in the list.

        Args:
            requirements_list: List of requirement dictionaries or Requirements objects
            objects: List of objects to check

        Returns:
            List of matching objects
        """
        matching_objects = []

        for obj in objects:
            if self.matches_any_requirements(obj, requirements_list):
                matching_objects.append(obj)

        return matching_objects

    def generate_random_objects(self, count: int = 10) -> List[Dict]:
        """
        Generate a list of objects with random attribute combinations.

        Args:
            count: Number of random objects to generate

        Returns:
            List of objects with random attributes
        """
        random_objects = []

        for _ in range(count):
            # Generate a random object with random attributes
            random_obj = {
                "shape_type": random.choice(SHAPE_TYPES),
                "color1": random.choice(COLORS),
                "color2": random.choice(COLORS),
                "size": random.choice(SIZES),
                "style": random.choice(STYLES)
            }

            # Ensure color2 is different for border/half styles
            if random_obj["style"] in ["border", "half"] and random_obj["color1"] == random_obj["color2"]:
                possible_colors = [c for c in COLORS if c != random_obj["color1"]]
                if possible_colors:
                    random_obj["color2"] = random.choice(possible_colors)

            random_objects.append(random_obj)

        return random_objects

    def generate_bfs_expressions(self, objects_data, bfs_pattern_type="all", samples_per_pattern=1,
                                bfs_ratio_single_attr=0.25, bfs_ratio_two_attr=0.25,
                                bfs_ratio_three_attr=0.25, bfs_ratio_four_attr=0.25,
                                num_expressions=None):
        """
        Generate BFS (attribute-based) referring expressions.

        Args:
            objects_data: List of object dictionaries
            bfs_pattern_type: Strategy for generating expressions
                               "all" - Use all attributes of objects
                               "patterns" - Use predefined patterns
            samples_per_pattern: Number of samples to generate per pattern
            bfs_ratio_single_attr: Proportion of single attribute expressions (default: 0.25)
            bfs_ratio_two_attr: Proportion of two attribute combinations (default: 0.25)
            bfs_ratio_three_attr: Proportion of three attribute combinations (default: 0.25)
            bfs_ratio_four_attr: Proportion of all attribute expressions (default: 0.25)
            num_expressions: If provided, limits the total number of expressions returned to this value

        Returns:
            List of expression dictionaries with exactly num_expressions items if specified
        """
        if not objects_data:
            return []

        # Validate sampling strategy
        valid_strategies = ["all", "patterns"]
        if bfs_pattern_type not in valid_strategies:
            raise ValueError(f"Invalid sampling strategy. Choose from: {valid_strategies}")

        # Check if we're using ratio-based sampling
        using_ratio_sampling = any(x != 0.25 for x in [
            bfs_ratio_single_attr, bfs_ratio_two_attr,
            bfs_ratio_three_attr, bfs_ratio_four_attr
        ])

        # Normalize ratios if they don't sum to 1
        if using_ratio_sampling:
            total_ratio = sum([bfs_ratio_single_attr, bfs_ratio_two_attr,
                               bfs_ratio_three_attr, bfs_ratio_four_attr])
            if total_ratio != 1.0:
                bfs_ratio_single_attr /= total_ratio
                bfs_ratio_two_attr /= total_ratio
                bfs_ratio_three_attr /= total_ratio
                bfs_ratio_four_attr /= total_ratio

        all_expressions = []

        # For each object, generate expressions based on sampling strategy
        all_expressions = []
        remaining_expressions = num_expressions if num_expressions is not None else None

        # Keep generating until we have enough expressions or run out of objects
        while remaining_expressions is None or remaining_expressions > 0:
            for obj in objects_data:
                # Create object info for expression generation
                obj_info = {
                    "shape_type": obj.get("shape_type"),
                    "color1": obj.get("color1"),
                    "color2": obj.get("color2"),
                    "size": obj.get("size"),
                    "style": obj.get("style")
                }

                # Skip objects that don't have required attributes
                if not all(k in obj_info and obj_info[k] is not None for k in ["shape_type", "color1", "style", "size"]):
                    continue

                # Get object style
                style = obj_info["style"]

                if bfs_pattern_type == "patterns":
                    obj_expressions = []

                    if using_ratio_sampling:
                        # Calculate samples per complexity group based on ratios
                        total_samples = samples_per_pattern * (len(self.single_attr_patterns) +
                                                              len(self.two_attr_patterns) +
                                                              len(self.three_attr_patterns) +
                                                              len(self.all_attr_patterns))

                        # Calculate samples per group
                        single_samples = int(total_samples * bfs_ratio_single_attr)
                        two_samples = int(total_samples * bfs_ratio_two_attr)
                        three_samples = int(total_samples * bfs_ratio_three_attr)
                        all_samples = int(total_samples * bfs_ratio_four_attr)

                        # Generate expressions for each complexity group based on ratios
                        ratio_data = [
                            (self.single_attr_patterns, single_samples),
                            (self.two_attr_patterns, two_samples),
                            (self.three_attr_patterns, three_samples),
                            (self.all_attr_patterns, all_samples)
                        ]

                        for pattern_list, num_samples in ratio_data:
                            if num_samples <= 0:
                                continue

                            # If we have more samples than patterns, distribute evenly
                            patterns_to_use = pattern_list

                            if num_samples < len(pattern_list):
                                # Randomly sample patterns if we need fewer than available
                                patterns_to_use = random.sample(pattern_list, num_samples)
                                samples_per_selected_pattern = 1
                            else:
                                # Distribute samples across patterns
                                samples_per_selected_pattern = max(1, num_samples // len(pattern_list))

                            # Generate expressions for selected patterns
                            for pattern_key in patterns_to_use:
                                pattern_expressions = self._generate_expressions_for_pattern(
                                    pattern_key, obj_info, style, samples_per_selected_pattern)
                                obj_expressions.extend(pattern_expressions)
                    else:
                        # Original behavior - sample each pattern equally
                        for template_key in list(self.template_patterns.keys()):
                            pattern_expressions = self._generate_expressions_for_pattern(
                                template_key, obj_info, style, samples_per_pattern)
                            obj_expressions.extend(pattern_expressions)

                    all_expressions.extend(obj_expressions)

                else:  # "all" strategy - use every available template
                    for template_key in list(self.template_patterns.keys()):
                        # Get templates for this key
                        if isinstance(self.template_patterns[template_key], dict):
                            if style not in self.template_patterns[template_key]:
                                continue
                            templates = self.template_patterns[template_key][style]
                        else:
                            templates = self.template_patterns[template_key]

                        # Use all templates
                        for template_tuple in templates:
                            template, requirements_list = template_tuple
                            expr_dict = self._create_expression(template_key, template, requirements_list, obj_info)
                            if expr_dict:
                                all_expressions.append(expr_dict)

                # Check if we've reached the required number of expressions
                if remaining_expressions is not None:
                    remaining_expressions = max(0, remaining_expressions - len(all_expressions))
                    if remaining_expressions <= 0:
                        break

            # If we're not limited by num_expressions, break after one pass
            if remaining_expressions is None:
                break

        # If num_expressions is specified, randomly sample exactly that many expressions
        if num_expressions is not None and all_expressions:
            # Ensure we don't ask for more expressions than available
            num_to_sample = min(num_expressions, len(all_expressions))
            random.shuffle(all_expressions)
            all_expressions = all_expressions[:num_to_sample]

        return all_expressions

    def _generate_expressions_for_pattern(self, template_key, obj_info, style, num_samples):
        """
        Helper method to generate expressions for a specific pattern.

        Args:
            template_key: The pattern key
            obj_info: Object attributes
            style: Object style
            num_samples: Number of samples to generate

        Returns:
            List of expression dictionaries
        """
        expressions = []

        # Get templates for this key
        if isinstance(self.template_patterns[template_key], dict):
            if style not in self.template_patterns[template_key]:
                return []
            templates = self.template_patterns[template_key][style]
        else:
            templates = self.template_patterns[template_key]

        # Sample N templates or all if fewer than N are available
        sample_count = min(num_samples, len(templates))
        if sample_count == 0:
            return []

        sampled_templates = random.sample(templates, sample_count)

        # Generate expressions for each sampled template
        for template_tuple in sampled_templates:
            template, requirements_list = template_tuple
            expr_dict = self._create_expression(template_key, template, requirements_list, obj_info)
            if expr_dict:
                expressions.append(expr_dict)

        return expressions

    def _create_expression(self, _, template, pre_computed_requirements, obj_info):
        """
        Create a single expression dictionary from a template.

        Args:
            template_key: The key of the template pattern
            template: The template string
            pre_computed_requirements: List of pre-computed Requirements objects
            obj_info: Object attributes dictionary

        Returns:
            Expression dictionary or None if creation fails
        """
        try:
            # Format the template
            format_dict = {
                "shape_type": obj_info.get("shape_type"),
                "size": obj_info.get("size"),
                "color1": obj_info.get("color1"),
                "color2": obj_info.get("color2")
            }

            expression = template.format(**format_dict)
            # Fix spacing for incomplete templates
            expression = ' '.join(expression.split())

            # Create concrete requirements by substituting placeholders with actual values
            concrete_requirements = []
            for req in pre_computed_requirements:
                concrete_req = {}

                if req.shape_type:
                    concrete_req["shape_type"] = obj_info.get("shape_type") if req.shape_type == "{shape_type}" else req.shape_type

                if req.size:
                    concrete_req["size"] = obj_info.get("size") if req.size == "{size}" else req.size

                if req.style:
                    concrete_req["style"] = req.style  # Style is already concrete

                if req.color1:
                    concrete_req["color1"] = obj_info.get("color1") if req.color1 == "{color1}" else (
                        obj_info.get("color2") if req.color1 == "{color2}" else req.color1
                    )

                if req.color2:
                    concrete_req["color2"] = obj_info.get("color2") if req.color2 == "{color2}" else (
                        obj_info.get("color1") if req.color2 == "{color1}" else req.color2
                    )

                concrete_requirements.append(concrete_req)

            return {
                "referring_expression": expression,
                "expression_type": "BFS",
                "target_requirements": concrete_requirements
            }

        except KeyError as e:
            # If formatting fails, return None
            print(f"Error formatting template: {template} with {obj_info}. Error: {e}")
            return None

# If run as a script, show some sample expressions
if __name__ == "__main__":
    handler = BFSExpressionHandler()

    print("=== BFS Expression Handler Demo ===")
    print("\nTemplate patterns:")
    for pattern, templates in handler.template_patterns.items():
        if isinstance(templates, dict):
            print(f"- {pattern}: {sum(len(t) for t in templates.values())} templates (style-specific)")
        else:
            print(f"- {pattern}: {len(templates)} templates")

    print("\nPattern groups:")
    for group_name, patterns in handler.pattern_groups.items():
        print(f"- {group_name}: {patterns}")

    print("\nSample BFS expressions with equal ratios:")
    # Create sample objects for demonstration
    sample_objects = [
        {
            "shape_type": "circle",
            "color1": "red",
            "color2": "blue",
            "size": "small",
            "style": "border"
        },
        {
            "shape_type": "square",
            "color1": "green",
            "color2": "yellow",
            "size": "big",
            "style": "solid"
        },
        {
            "shape_type": "triangle",
            "color1": "purple",
            "color2": "orange",
            "size": "small",
            "style": "half"
        }
    ]

    # Test the patterns strategy that samples each pattern once
    expressions = handler.generate_bfs_expressions(
        sample_objects,
        bfs_pattern_type="patterns",
        samples_per_pattern=1
    )
    for i, expr in enumerate(expressions[:5], 1):
        print(f"{i}. {expr['referring_expression']}")

    print("\nSample BFS expressions with custom ratios:")
    # Test with custom ratios
    expressions = handler.generate_bfs_expressions(
        sample_objects,
        bfs_pattern_type="patterns",
        samples_per_pattern=1,
        bfs_ratio_single_attr=0.6,
        bfs_ratio_two_attr=0.3,
        bfs_ratio_three_attr=0.1,
        bfs_ratio_four_attr=0.0
    )
    for i, expr in enumerate(expressions[:5], 1):
        print(f"{i}. {expr['referring_expression']}")
        # Print all possible requirement sets
        print("   Requirements:")
        for j, req_set in enumerate(expr['target_requirements'], 1):
            if len(expr['target_requirements']) > 1:
                print(f"    Option {j}: {', '.join([f'{k}={v}' for k, v in req_set.items()])}")
            else:
                print(f"    {', '.join([f'{k}={v}' for k, v in req_set.items()])}")
        print()
