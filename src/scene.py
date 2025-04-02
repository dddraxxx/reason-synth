from PIL import Image, ImageDraw
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import json
import math

from .shapes import Shape

class Scene:
    def __init__(
        self,
        grid_size: Tuple[int, int],
        image_size: Tuple[int, int] = (800, 600),
        background_color: Tuple[int, int, int] = (240, 240, 240),
        region_ratio_range: Tuple[float, float] = (0.5, 0.8),  # Min and max ratio of image to use for grid
        max_offset_ratio: float = 0.2,  # Maximum offset as a ratio of cell size
    ):
        """
        Initialize a scene with a grid of shapes.

        Args:
            grid_size: Size of the grid (rows, columns)
            image_size: Size of the output image (width, height)
            background_color: Background color of the image
            region_ratio_range: Range of ratios for random region size (min, max)
            max_offset_ratio: Maximum offset as a ratio of cell size
        """
        self.grid_size = grid_size
        self.image_size = image_size
        self.background_color = background_color
        self.region_ratio_range = region_ratio_range
        self.shapes: List[Shape] = []
        self.grid_positions: List[Tuple[int, int]] = []
        self.image = Image.new('RGB', image_size, background_color)
        self.min_distance_ratio = 0.25  # Minimum distance as ratio of bbox size

        # Select a random region size for the grid
        min_ratio, max_ratio = region_ratio_range
        width_ratio = np.random.uniform(min_ratio, max_ratio)
        height_ratio = np.random.uniform(min_ratio, max_ratio)

        # Calculate region dimensions
        region_width = int(image_size[0] * width_ratio)
        region_height = int(image_size[1] * height_ratio)

        # Calculate random position for region
        max_x_offset = image_size[0] - region_width
        max_y_offset = image_size[1] - region_height
        region_x = np.random.randint(0, max_x_offset + 1) if max_x_offset > 0 else 0
        region_y = np.random.randint(0, max_y_offset + 1) if max_y_offset > 0 else 0

        # Calculate grid cell size
        self.cell_width = region_width // grid_size[1]
        self.cell_height = region_height // grid_size[0]

        # Calculate maximum offset to avoid overlap
        # Use half the cell size as maximum possible offset
        self.max_offset_x = int(self.cell_width * max_offset_ratio)
        self.max_offset_y = int(self.cell_height * max_offset_ratio)

        # Store region information
        self.region = (region_x, region_y, region_width, region_height)

    def _distance_between_centers(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_min_required_distance(self, shape1: Shape, shape2: Shape) -> float:
        """Get minimum required distance between two shapes based on their bounding boxes."""
        # Get bounding box dimensions
        bbox1 = shape1.bbox
        bbox2 = shape2.bbox

        # Calculate width and height of each bbox
        width1 = bbox1[2] - bbox1[0]
        height1 = bbox1[3] - bbox1[1]
        width2 = bbox2[2] - bbox2[0]
        height2 = bbox2[3] - bbox2[1]

        # Calculate the diameter of each shape's bounding box
        # For rotated shapes, this gives a better approximation
        diameter1 = math.sqrt(width1**2 + height1**2)
        diameter2 = math.sqrt(width2**2 + height2**2)

        # Use the smaller object for the calculation (its bbox diagonal)
        min_diameter = min(diameter1, diameter2)

        # Return 1/4 of the minimum diameter
        return min_diameter * self.min_distance_ratio

    def _is_position_valid(self, shape: Shape, position: Tuple[int, int]) -> bool:
        """Check if a position is valid for a shape based on minimum distance constraint."""
        # Create a temporary copy of the shape with the new position
        temp_shape = Shape(
            shape_type=shape.shape_type,
            size=shape.size,
            color1=shape.color1,
            color2=shape.color2,
            style=shape.style,
            position=position,
            rotation=shape.rotation
        )
        # Calculate the bounding box with rotation accounted for
        temp_shape.bbox = temp_shape._calculate_bbox()

        # Check against all existing shapes
        for existing_shape in self.shapes:
            # Calculate the required minimum distance
            min_distance = self._get_min_required_distance(temp_shape, existing_shape)

            # Calculate actual distance between centers
            actual_distance = self._distance_between_centers(position, existing_shape.position)

            # If too close, position is invalid
            if actual_distance < min_distance:
                return False

        return True

    def add_shape(self, shape: Shape, grid_pos: Tuple[int, int]) -> None:
        """
        Add a shape to the scene at the specified grid position.

        Args:
            shape: Shape object to add
            grid_pos: Grid position (row, col)
        """
        row, col = grid_pos
        region_x, region_y, _, _ = self.region

        # Calculate base position in pixels (center of the cell)
        base_x = region_x + col * self.cell_width + self.cell_width // 2
        base_y = region_y + row * self.cell_height + self.cell_height // 2

        # If this is a rotated shape, we need to ensure it's properly placed
        is_rotated = shape.rotation != 0 and shape.shape_type != "circle"

        # Try to find a valid position with the minimum distance constraint
        max_attempts = 20  # Limit attempts to avoid infinite loops
        valid_position_found = False

        for attempt in range(max_attempts):
            # Apply random offset (constrained to avoid overlap)
            # For rotated shapes, we reduce the maximum offset to account for larger bounding boxes
            offset_factor = 0.8 if is_rotated else 1.0
            max_offset_x = int(self.max_offset_x * offset_factor)
            max_offset_y = int(self.max_offset_y * offset_factor)

            offset_x = np.random.randint(-max_offset_x, max_offset_x + 1)
            offset_y = np.random.randint(-max_offset_y, max_offset_y + 1)

            # Calculate new position
            new_position = (base_x + offset_x, base_y + offset_y)

            # Check if this position is valid
            if len(self.shapes) == 0 or self._is_position_valid(shape, new_position):
                # Position is valid, set it and break the loop
                shape.position = new_position
                valid_position_found = True
                break

        # If no valid position found after max attempts, use base position
        if not valid_position_found:
            shape.position = (base_x, base_y)

        # Update shape's bounding box with new position
        shape.bbox = shape._calculate_bbox()

        # Add to scene
        self.shapes.append(shape)
        self.grid_positions.append(grid_pos)

    def render(self) -> Image.Image:
        """Render all shapes in the scene and return the image."""
        # Create a fresh image
        self.image = Image.new('RGBA', self.image_size, self.background_color + (255,))

        # Optionally draw the region borders for debugging
        # draw = ImageDraw.Draw(self.image)
        # region_x, region_y, region_width, region_height = self.region
        # draw.rectangle(
        #     (region_x, region_y, region_x + region_width, region_y + region_height),
        #     outline=(200, 200, 200, 255), width=2
        # )

        # Draw all shapes
        for shape in self.shapes:
            shape.draw(self.image)

        # Convert to RGB for saving as JPEG if needed
        rgb_image = Image.new('RGB', self.image_size, self.background_color)
        rgb_image.paste(self.image, (0, 0), self.image)

        return rgb_image

    def save_image(self, file_path: str) -> None:
        """Save the rendered scene to a file."""
        # Create directory if needed and if dirname is not empty
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        image = self.render()
        image.save(file_path)

    def to_dict(self) -> Dict:
        """Convert the scene to a dictionary for annotation."""
        return {
            "grid_size": self.grid_size,
            "image_size": self.image_size,
            "region": self.region,
            "shapes": [
                {
                    **shape.to_dict(),
                    "grid_position": self.grid_positions[i]
                }
                for i, shape in enumerate(self.shapes)
            ]
        }

    def save_annotation(self, file_path: str, image_path: str) -> None:
        """
        Save the scene annotation to a JSON file.

        Args:
            file_path: Path to save the annotation JSON
            image_path: Path to the image file (to include in annotation)
        """
        # Create directory if needed and if dirname is not empty
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Create annotation dictionary
        annotation = {
            "image_path": image_path,
            "scene": self.to_dict()
        }

        # Save as JSON
        with open(file_path, 'w') as f:
            json.dump(annotation, f, indent=2)

    def save_to_jsonl(self, jsonl_file: str, image_path: str) -> None:
        """
        Append scene annotation to a JSONL file.

        Args:
            jsonl_file: Path to the JSONL file
            image_path: Path to the image file
        """
        # Create directory if needed and if dirname is not empty
        dirname = os.path.dirname(jsonl_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Create annotation dictionary
        annotation = {
            "image_path": image_path,
            "scene": self.to_dict()
        }

        # Append to JSONL file
        with open(jsonl_file, 'a') as f:
            f.write(json.dumps(annotation) + '\n')
