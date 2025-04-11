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
        cell_size: int = 70, # NEW: Base cell size
        region_variation_ratio: float = 0.1, # NEW: Random variation ratio for region
        max_offset_ratio: float = 0.2,  # Maximum offset as a ratio of cell size
        max_overlap_ratio: float = 0.1  # Renamed, default allows slight overlap
    ):
        """
        Initialize a scene with a grid of shapes.

        Args:
            grid_size: Size of the grid (rows, columns)
            image_size: Size of the output image (width, height)
            background_color: Background color of the image
            cell_size: Base size of a grid cell in pixels.
            region_variation_ratio: Random variation (+/-) applied to the total region size calculated from cell_size and grid_size.
            max_offset_ratio: Maximum offset as a ratio of cell size
            max_overlap_ratio: Max allowed overlap distance as a ratio of the smaller shape's bbox diagonal.
                               0 means bounding circles just touch, >0 allows overlap.
        """
        self.grid_size = grid_size
        self.image_size = image_size
        self.background_color = background_color
        self.max_overlap_ratio = max_overlap_ratio
        self.shapes: List[Shape] = []
        self.grid_positions: List[Tuple[int, int]] = []
        self.image = Image.new('RGB', image_size, background_color)

        # Add placement tracking stats
        self.total_shapes_placed = 0
        self.successful_placements = 0
        self.total_attempts = 0

        rows, cols = grid_size
        img_width, img_height = image_size

        # Calculate base region size from cell size and grid
        base_region_width = cols * cell_size
        base_region_height = rows * cell_size

        # Apply random variation
        min_scale = 1.0 - region_variation_ratio
        max_scale = 1.0 + region_variation_ratio
        scale_factor = np.random.uniform(min_scale, max_scale)

        region_width = int(base_region_width * scale_factor)
        region_height = int(base_region_height * scale_factor)

        # Clamp region size to image boundaries
        region_width = min(region_width, img_width)
        region_height = min(region_height, img_height)

        # Calculate random position for region
        max_x_offset = image_size[0] - region_width
        max_y_offset = image_size[1] - region_height
        region_x = np.random.randint(0, max_x_offset + 1) if max_x_offset > 0 else 0
        region_y = np.random.randint(0, max_y_offset + 1) if max_y_offset > 0 else 0

        # Calculate actual grid cell size based on the final region size
        self.cell_width = region_width // cols if cols > 0 else 0
        self.cell_height = region_height // rows if rows > 0 else 0

        # Calculate maximum offset based on the *actual* cell size
        self.max_offset_x = int(self.cell_width * max_offset_ratio)
        self.max_offset_y = int(self.cell_height * max_offset_ratio)

        # Store region information
        self.region = (region_x, region_y, region_width, region_height)

    def _distance_between_centers(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _get_min_required_distance(self, shape1: Shape, shape2: Shape) -> float:
        """Calculate the minimum required center distance based on max_overlap_ratio."""
        bbox1 = shape1.bbox
        bbox2 = shape2.bbox
        if not bbox1 or not bbox2:
            # Handle cases where bbox might not be calculated yet (shouldn't happen in normal flow)
            return 0

        width1 = bbox1[2] - bbox1[0]
        height1 = bbox1[3] - bbox1[1]
        width2 = bbox2[2] - bbox2[0]
        height2 = bbox2[3] - bbox2[1]

        diameter1 = math.sqrt(width1**2 + height1**2)
        diameter2 = math.sqrt(width2**2 + height2**2)

        # Use diameters as approximations for radii
        radius1 = diameter1 / 2.0
        radius2 = diameter2 / 2.0

        # The distance if bounding circles just touch
        touching_distance = radius1 + radius2

        # The maximum allowed overlap distance, scaled by the smaller shape
        smaller_bbox_diagonal = min(diameter1, diameter2)
        allowed_overlap_distance = smaller_bbox_diagonal * self.max_overlap_ratio

        # Minimum required distance = touching distance - allowed overlap
        min_req_dist = touching_distance - allowed_overlap_distance

        # Distance cannot be negative
        return max(0, min_req_dist)

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

        self.total_shapes_placed += 1
        attempts_for_this_shape = 0

        for attempt in range(max_attempts):
            attempts_for_this_shape += 1
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
                self.successful_placements += 1
                break

        # Update total attempts stats
        self.total_attempts += attempts_for_this_shape

        # If no valid position found after max attempts, use base position
        if not valid_position_found:
            if os.environ.get('dp'):
                print(f"\nFailed to find valid position for shape {shape.shape_type} at {grid_pos}, using base position\n")
            shape.position = (base_x, base_y)
        else:
            if os.environ.get('dp'):
                print(f"Found valid position for shape {shape.shape_type} at {grid_pos}")

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

    def get_placement_stats(self) -> Dict:
        """Get statistics about shape placement attempts."""
        if self.total_shapes_placed == 0:
            success_rate = 0.0
        else:
            success_rate = (self.successful_placements / self.total_shapes_placed) * 100

        if self.total_shapes_placed == 0:
            avg_attempts = 0.0
        else:
            avg_attempts = self.total_attempts / self.total_shapes_placed

        return {
            "total_shapes": self.total_shapes_placed,
            "successful_placements": self.successful_placements,
            "failed_placements": self.total_shapes_placed - self.successful_placements,
            "success_rate": success_rate,
            "total_attempts": self.total_attempts,
            "avg_attempts_per_shape": avg_attempts
        }
