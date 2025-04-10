"""
BFS Expression Handler module for Reason-Synth dataset.

This module provides a class that handles both the generation and matching
of BFS (attribute-based) referring expressions.
"""

import copy
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
import argparse

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
                                num_expressions=None, reasoning_level: Optional[int] = None):
        """
        Generate BFS (attribute-based) referring expressions.

        Args:
            objects_data: List of object dictionaries
            bfs_pattern_type: Strategy for generating expressions. Options are:
                - "all": Generate expressions using all available templates
                - "patterns": Generate expressions using patterns (each pattern contains multiple templates)
            samples_per_pattern: Number of samples per pattern (used in "patterns" mode)
            bfs_ratio_single_attr: Proportion of single attribute expressions (if ratios used)
            bfs_ratio_two_attr: Proportion of two attribute combinations (if ratios used)
            bfs_ratio_three_attr: Proportion of three attribute combinations (if ratios used)
            bfs_ratio_four_attr: Proportion of all attribute expressions (if ratios used)
            num_expressions: Limit the total number of expressions returned.
            reasoning_level: If specified, only generate expressions with this reasoning level.
                           When specified, ratio/pattern arguments are ignored.

        Returns:
            List of expression dictionaries.
        """
        if not objects_data:
            return []

        # --- New: Reasoning Level Specific Generation ---
        if reasoning_level is not None:
            if num_expressions is None:
                raise ValueError("num_expressions must be specified when reasoning_level is set.")

            generated_expressions = []
            max_attempts = num_expressions * 100 # Safety break to prevent infinite loops
            attempts = 0

            # Collect all available templates first
            all_template_tuples = []
            for key, templates_or_dict in self.template_patterns.items():
                if isinstance(templates_or_dict, dict): # Style-specific
                    for style, templates in templates_or_dict.items():
                        for t_tuple in templates:
                            all_template_tuples.append((key, t_tuple[0], t_tuple[1])) # (key, template, requirements)
                else: # Non-style-specific
                    for t_tuple in templates_or_dict:
                        all_template_tuples.append((key, t_tuple[0], t_tuple[1]))

            if not all_template_tuples:
                return [] # No templates available

            while len(generated_expressions) < num_expressions and attempts < max_attempts:
                attempts += 1
                # Randomly pick an object and a template
                obj = random.choice(objects_data)
                template_key, template, pre_computed_reqs = random.choice(all_template_tuples)

                # Basic object info check
                obj_info = {
                    "shape_type": obj.get("shape_type"),
                    "color1": obj.get("color1"),
                    "color2": obj.get("color2"),
                    "size": obj.get("size"),
                    "style": obj.get("style")
                }
                if obj_info["style"] is None: continue # Need style for template matching

                # Check if the template is style-specific and matches the object's style
                is_style_specific_template = isinstance(self.template_patterns.get(template_key), dict)
                if is_style_specific_template and obj_info["style"] not in self.template_patterns[template_key]:
                    continue # Style mismatch

                # Try to create the expression with the target reasoning level
                expr_dict = self._create_expression(
                    template_key, template, pre_computed_reqs, obj_info,
                    target_reasoning_level=reasoning_level
                )

                if expr_dict:
                    # Check for duplicates before adding (optional, but good practice)
                    # Simple check based on the expression string
                    if expr_dict["referring_expression"] not in [e["referring_expression"] for e in generated_expressions]:
                        generated_expressions.append(expr_dict)

            if len(generated_expressions) < num_expressions:
                print(f"Warning: Only generated {len(generated_expressions)}/{num_expressions} expressions for reasoning level {reasoning_level}. "
                      f"Try increasing max_attempts or check object/template variety.")

            return generated_expressions
        # --- End: Reasoning Level Specific Generation ---

        # --- Original Logic (Ratios/Patterns/All) ---
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
            if total_ratio > 0 and abs(total_ratio - 1.0) > 1e-6: # Allow for floating point inaccuracies
                bfs_ratio_single_attr /= total_ratio
                bfs_ratio_two_attr /= total_ratio
                bfs_ratio_three_attr /= total_ratio
                bfs_ratio_four_attr /= total_ratio
            elif total_ratio == 0:
                 raise ValueError("All BFS ratios are zero. Cannot generate expressions.")

        all_expressions = []
        # Keep track of how many expressions we need based on num_expressions
        remaining_expressions_needed = num_expressions if num_expressions is not None else float('inf')

        # Iterate through objects to generate expressions until count is met or objects run out
        # We might need multiple passes if num_expressions is large
        objects_shuffled = objects_data[:] # Create a copy to shuffle
        generation_complete = False
        while remaining_expressions_needed > 0 and not generation_complete:
            random.shuffle(objects_shuffled)
            expressions_in_pass = 0
            for obj in objects_shuffled:
                if remaining_expressions_needed <= 0: break

                obj_info = {
                    "shape_type": obj.get("shape_type"),
                    "color1": obj.get("color1"),
                    "color2": obj.get("color2"),
                    "size": obj.get("size"),
                    "style": obj.get("style")
                }

                if not all(k in obj_info and obj_info[k] is not None for k in ["shape_type", "color1", "style", "size"]):
                    continue

                style = obj_info["style"]
                generated_for_this_object = []

                if bfs_pattern_type == "patterns":
                    if using_ratio_sampling:
                        # Determine total potential patterns for this object's style
                        potential_patterns = []
                        if self.single_attr_patterns: potential_patterns.extend(self.single_attr_patterns)
                        if self.two_attr_patterns: potential_patterns.extend(self.two_attr_patterns)
                        if self.three_attr_patterns: potential_patterns.extend(self.three_attr_patterns)
                        if self.all_attr_patterns: potential_patterns.extend(self.all_attr_patterns)

                        # Adjust total_samples based on potentially fewer available patterns for the object's style
                        applicable_patterns_count = 0
                        for pattern_key in potential_patterns:
                            if isinstance(self.template_patterns.get(pattern_key), dict):
                                if style in self.template_patterns[pattern_key]: applicable_patterns_count += 1
                            else:
                                applicable_patterns_count += 1

                        # Estimate total expressions we *could* generate for this object with 1 sample per applicable pattern
                        total_potential_samples = applicable_patterns_count * samples_per_pattern
                        if total_potential_samples == 0: continue # Skip if object style has no matching patterns

                        # Calculate samples per complexity group based on ratios applied to potential samples
                        single_samples = round(total_potential_samples * bfs_ratio_single_attr)
                        two_samples = round(total_potential_samples * bfs_ratio_two_attr)
                        three_samples = round(total_potential_samples * bfs_ratio_three_attr)
                        all_samples = round(total_potential_samples * bfs_ratio_four_attr)
                        # Ensure we generate at least samples_per_pattern total if ratios are low
                        total_target_samples = max(samples_per_pattern, single_samples + two_samples + three_samples + all_samples)

                        ratio_data = [
                            (self.single_attr_patterns, single_samples),
                            (self.two_attr_patterns, two_samples),
                            (self.three_attr_patterns, three_samples),
                            (self.all_attr_patterns, all_samples)
                        ]

                        temp_expressions = []
                        for pattern_list, num_samples_target in ratio_data:
                            if num_samples_target <= 0: continue
                            # Generate slightly more than needed initially, will sample later
                            temp_expressions.extend(self._generate_expressions_for_pattern(
                                pattern_list, obj_info, style, num_samples_target * 2 # Generate more to allow sampling
                            ))

                        # Sample from the generated expressions to meet the target count for this object
                        random.shuffle(temp_expressions)
                        generated_for_this_object = temp_expressions[:total_target_samples]

                    else: # Original behavior - sample each pattern equally
                        for template_key in self._get_applicable_patterns(style):
                            pattern_expressions = self._generate_expressions_for_pattern(
                                [template_key], obj_info, style, samples_per_pattern
                            )
                            generated_for_this_object.extend(pattern_expressions)

                else: # "all" strategy - use every available template for the object's style
                    for template_key in self._get_applicable_patterns(style):
                        # Get templates for this key
                        if isinstance(self.template_patterns[template_key], dict):
                            templates = self.template_patterns[template_key].get(style, [])
                        else:
                            templates = self.template_patterns[template_key]

                        for template_tuple in templates:
                            template, requirements_list = template_tuple
                            # Pass target_reasoning_level=None for default behavior
                            expr_dict = self._create_expression(template_key, template, requirements_list, obj_info, target_reasoning_level=None)
                            if expr_dict:
                                generated_for_this_object.append(expr_dict)

                # Add unique expressions generated for this object, respecting num_expressions limit
                added_count = 0
                random.shuffle(generated_for_this_object)
                for expr in generated_for_this_object:
                    if remaining_expressions_needed <= 0: break
                    # More robust check for duplicates based on expression and target requirements
                    is_duplicate = False
                    for existing_expr in all_expressions:
                        if expr["referring_expression"] == existing_expr["referring_expression"] and \
                           expr["target_requirements"] == existing_expr["target_requirements"]:
                           is_duplicate = True
                           break
                    if not is_duplicate:
                        all_expressions.append(expr)
                        remaining_expressions_needed -= 1
                        added_count += 1
                expressions_in_pass += added_count

            # If a full pass over objects didn't add any new expressions, assume we can't generate more
            if expressions_in_pass == 0:
                generation_complete = True

        # Final check: If num_expressions was specified but we couldn't reach it
        if num_expressions is not None and len(all_expressions) < num_expressions:
             print(f"Warning: Could only generate {len(all_expressions)}/{num_expressions} unique BFS expressions with the given settings.")

        # Ensure we return exactly num_expressions if specified (already handled by remaining_expressions_needed logic)
        return all_expressions

    def _get_applicable_patterns(self, style):
        """Returns a list of pattern keys applicable to the given style."""
        applicable = []
        for key, templates_or_dict in self.template_patterns.items():
            if isinstance(templates_or_dict, dict):
                if style in templates_or_dict:
                    applicable.append(key)
            else:
                applicable.append(key)
        return applicable

    def _generate_expressions_for_pattern(self, pattern_keys, obj_info, style, num_samples):
        """
        Helper method to generate expressions for specific pattern keys.

        Args:
            pattern_keys: List of pattern keys
            obj_info: Object attributes
            style: Object style
            num_samples: Target number of samples across all keys

        Returns:
            List of expression dictionaries
        """
        expressions = []
        available_template_tuples = []

        # Collect all applicable templates for the given keys and style
        for template_key in pattern_keys:
            templates_or_dict = self.template_patterns.get(template_key)
            if templates_or_dict is None: continue

            if isinstance(templates_or_dict, dict):
                templates = templates_or_dict.get(style, [])
            else:
                templates = templates_or_dict

            for template_tuple in templates:
                available_template_tuples.append((template_key, template_tuple))

        if not available_template_tuples: return []

        # Sample N templates or all if fewer than N are available
        sample_count = min(num_samples, len(available_template_tuples))
        sampled_items = random.sample(available_template_tuples, sample_count)

        # Generate expressions for each sampled template
        for template_key, template_tuple in sampled_items:
            template, requirements_list = template_tuple
            # Pass target_reasoning_level=None for default behavior
            expr_dict = self._create_expression(template_key, template, requirements_list, obj_info, target_reasoning_level=None)
            if expr_dict:
                expressions.append(expr_dict)

        return expressions

    def _create_expression(self, template_key, template, pre_computed_requirements, obj_info, target_reasoning_level: Optional[int] = None):
        """
        Create a single expression dictionary from a template, potentially filtering by reasoning level.

        Args:
            template_key: The key of the template pattern
            template: The template string
            pre_computed_requirements: List of pre-computed Requirements objects
            obj_info: Object attributes dictionary
            target_reasoning_level: If specified, only return expressions matching this level.

        Returns:
            Expression dictionary or None if creation fails or reasoning level doesn't match.
        """
        try:
            # Create concrete requirements by substituting placeholders with actual values
            concrete_requirements = []
            for req in pre_computed_requirements:
                concrete_req = {}
                # Fill concrete_req based on req and obj_info (handling placeholders)
                if req.shape_type:
                    concrete_req["shape_type"] = obj_info.get("shape_type") if req.shape_type == "{shape_type}" else req.shape_type
                if req.size:
                    concrete_req["size"] = obj_info.get("size") if req.size == "{size}" else req.size
                if req.style:
                    concrete_req["style"] = req.style # Style is already concrete
                if req.color1:
                    concrete_req["color1"] = obj_info.get("color1") if req.color1 == "{color1}" else (
                        obj_info.get("color2") if req.color1 == "{color2}" else req.color1
                    )
                if req.color2:
                    concrete_req["color2"] = obj_info.get("color2") if req.color2 == "{color2}" else (
                        obj_info.get("color1") if req.color2 == "{color1}" else req.color2
                    )

                # Filter out None values before adding
                concrete_req_filtered = {k: v for k, v in concrete_req.items() if v is not None}
                if concrete_req_filtered: # Only add if there are actual requirements
                    concrete_requirements.append(concrete_req_filtered)

            # If no valid concrete requirements could be formed (e.g., object missing needed attributes), fail early.
            if not concrete_requirements:
                return None

            # Calculate reasoning level based on the maximum number of attributes in any single requirement option
            max_attrs = 0
            if concrete_requirements: # Ensure there's at least one requirement
                max_attrs = max(len(req) for req in concrete_requirements)

            actual_reasoning_level = max_attrs

            # --- Reasoning Level Check ---
            if target_reasoning_level is not None and actual_reasoning_level != target_reasoning_level:
                return None # Does not match the target level
            # --- End Check ---

            # Format the template (only if reasoning level matches or isn't specified)
            format_dict = {
                "shape_type": obj_info.get("shape_type"),
                "size": obj_info.get("size"),
                "color1": obj_info.get("color1"),
                "color2": obj_info.get("color2")
            }
            # Filter out None values from format_dict to avoid errors with templates expecting attributes the object doesn't have
            format_dict_filtered = {k: v for k, v in format_dict.items() if v is not None}

            expression = template.format(**format_dict_filtered)
            # Fix spacing for incomplete templates
            expression = ' '.join(expression.split())

            return {
                "referring_expression": expression,
                "expression_type": "BFS",
                "target_requirements": concrete_requirements,
                "reasoning_level": actual_reasoning_level # Use the calculated level
            }

        except KeyError as e:
            # If formatting fails (e.g., template placeholder missing in filtered format_dict), return None
            # print(f"Warning: Formatting failed for template '{template}' with object {obj_info}. Error: {e}")
            return None
        except Exception as e:
            # Catch other potential errors during requirement generation or formatting
            print(f"Error creating expression for template '{template}' with object {obj_info}. Error: {e}")
            return None

# If run as a script, show some sample expressions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BFS referring expression templates and test matching.")
    parser.add_argument("--count", type=int, default=10, help="Number of expressions to generate.")
    parser.add_argument("--reasoning_level", "-r", type=int, default=None, help="Specific reasoning level (number of attributes) to target.")
    parser.add_argument("--pattern_type", type=str, default="patterns", choices=["all", "patterns"], help="Strategy for expression generation.")
    parser.add_argument("--samples_per_pattern", type=int, default=1, help="Samples per pattern (if pattern_type='patterns').")
    parser.add_argument("--ratio1", type=float, default=0.25, help="Ratio for single attribute expressions.")
    parser.add_argument("--ratio2", type=float, default=0.25, help="Ratio for two attribute expressions.")
    parser.add_argument("--ratio3", type=float, default=0.25, help="Ratio for three attribute expressions.")
    parser.add_argument("--ratio4", type=float, default=0.25, help="Ratio for four attribute expressions.")

    args = parser.parse_args()

    handler = BFSExpressionHandler()

    print("=== BFS Expression Handler Demo ===")

    # Create sample objects for demonstration
    sample_objects = handler.generate_random_objects(count=20) # Generate more objects for better variety
    if not sample_objects:
        print("Error: Could not generate sample objects.")
        exit()

    print(f"Generated {len(sample_objects)} random sample objects.")

    if args.reasoning_level is not None:
        print(f"\nGenerating {args.count} expressions with reasoning level {args.reasoning_level}:")
        expressions = handler.generate_bfs_expressions(
            sample_objects,
            num_expressions=args.count,
            reasoning_level=args.reasoning_level
        )
    else:
        print(f"\nGenerating {args.count} expressions using '{args.pattern_type}' strategy:")
        if args.pattern_type == "patterns":
            print(f"  Samples per pattern: {args.samples_per_pattern}")
            print(f"  Ratios (1/2/3/4 attrs): {args.ratio1:.2f}/{args.ratio2:.2f}/{args.ratio3:.2f}/{args.ratio4:.2f}")

        expressions = handler.generate_bfs_expressions(
            sample_objects,
            bfs_pattern_type=args.pattern_type,
            samples_per_pattern=args.samples_per_pattern,
            bfs_ratio_single_attr=args.ratio1,
            bfs_ratio_two_attr=args.ratio2,
            bfs_ratio_three_attr=args.ratio3,
            bfs_ratio_four_attr=args.ratio4,
            num_expressions=args.count
        )

    print(f"\n--- Generated Expressions ({len(expressions)} total) ---")
    for i, expr in enumerate(expressions, 1):
        print(f"{i}. {expr['referring_expression']}")
        print(f"   Reasoning Level: {expr['reasoning_level']}")
        print("   Requirements:")
        for j, req_set in enumerate(expr['target_requirements'], 1):
            req_str = ', '.join([f'{k}={v}' for k, v in req_set.items()])
            if len(expr['target_requirements']) > 1:
                print(f"    Option {j}: {req_str}")
            else:
                print(f"    {req_str}")
        # Find matching objects (optional demo)
        matches = handler.find_matching_objects(expr['target_requirements'], sample_objects)
        print(f"   Matching Objects: {len(matches)}")
        print()
