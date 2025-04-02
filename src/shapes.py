from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple, Dict, List, Literal, Optional, Union
import os
import json
import math

# Load configuration from JSON file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "object_config.json")
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Define types
ShapeType = Literal["triangle", "square", "circle"]
SizeType = Literal["big", "small"]
StyleType = Literal["solid", "half", "border"]

# Define constants from config
COLORS = {k: tuple(v) for k, v in CONFIG["colors"].items()}
SIZE_PARAMS = CONFIG["size_params"]
ROTATION_RANGES = CONFIG["rotation_ranges"]

class Shape:
    def __init__(
        self,
        shape_type: ShapeType,
        size: SizeType,
        color1: str,
        color2: Optional[str] = None,
        style: StyleType = "solid",
        position: Tuple[int, int] = (0, 0),
        rotation: float = 0.0,
    ):
        """
        Initialize a shape object.

        Args:
            shape_type: Type of shape ("triangle", "square", "circle")
            size: Size of the shape ("big", "small")
            color1: Primary color of the shape
            color2: Secondary color (for half style or border)
            style: Style of the shape ("solid", "half", "border")
            position: Position of the shape's center (x, y)
            rotation: Rotation angle in degrees
        """
        self.shape_type = shape_type
        self.size = size
        self.color1 = color1
        self.color2 = color2 if color2 else color1
        self.style = style
        self.position = position
        self.rotation = rotation
        self.size_px = SIZE_PARAMS[size]
        self.bbox = self._calculate_bbox()

    def _get_rotated_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Rotate points around the shape's center position.

        Args:
            points: List of point coordinates (x, y)

        Returns:
            List of rotated point coordinates
        """
        if self.rotation == 0:
            return points

        x, y = self.position
        rad = np.radians(self.rotation)
        cos_val, sin_val = np.cos(rad), np.sin(rad)

        rotated_points = []
        for px, py in points:
            # Translate to origin, rotate, translate back
            nx = cos_val * (px - x) - sin_val * (py - y) + x
            ny = sin_val * (px - x) + cos_val * (py - y) + y
            rotated_points.append((nx, ny))

        return rotated_points

    def _calculate_bbox(self) -> Tuple[int, int, int, int]:
        """Calculate the bounding box of the shape, accounting for rotation."""
        x, y = self.position
        half_size = self.size_px // 2

        if self.shape_type == "triangle":
            # Height of equilateral triangle = side length * √3/2
            height = self.size_px * np.sqrt(3) / 2

            # Adjust for small triangles
            if self.size == "small":
                height *= 1.2

            # Define the triangle points
            points = [
                (x, y - height/2),  # top
                (x - half_size, y + height/2),  # bottom left
                (x + half_size, y + height/2),  # bottom right
            ]

            # Apply rotation if needed
            points = self._get_rotated_points(points)

        elif self.shape_type == "square":
            # Define corner points of the square
            points = [
                (x - half_size, y - half_size),  # top-left
                (x + half_size, y - half_size),  # top-right
                (x + half_size, y + half_size),  # bottom-right
                (x - half_size, y + half_size),  # bottom-left
            ]

            # Apply rotation if needed
            points = self._get_rotated_points(points)

        elif self.shape_type == "circle":
            # For circles, rotation doesn't affect the bounding box
            # Just return a square bounding box
            return (
                int(x - half_size),
                int(y - half_size),
                int(x + half_size),
                int(y + half_size)
            )

        # Calculate the minimum and maximum coordinates for the bounding box
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        # Add a small buffer for border styles
        if self.style == "border":
            buffer = 2
            min_x -= buffer
            min_y -= buffer
            max_x += buffer
            max_y += buffer

        # Return the bounding box as integer coordinates
        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def draw(self, image: Image.Image) -> None:
        """Draw the shape on the given image."""
        draw = ImageDraw.Draw(image)

        if self.shape_type == "triangle":
            self._draw_triangle(draw)
        elif self.shape_type == "square":
            self._draw_square(draw)
        elif self.shape_type == "circle":
            self._draw_circle(draw)

    def _create_half_color_mask(self, width, height, color1, color2):
        """Create a mask with two colors for half-color style."""
        mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)

        # Create a gradient image or pattern
        # Here we're using a simple left/right split
        draw.rectangle((0, 0, width//2, height), fill=color1)
        draw.rectangle((width//2, 0, width, height), fill=color2)

        return mask

    def _draw_triangle(self, draw: ImageDraw.ImageDraw) -> None:
        """Draw a triangle on the ImageDraw object."""
        x, y = self.position
        size = self.size_px
        half_size = size // 2

        # Define the triangle points - using equilateral triangle for better appearance
        # Height of equilateral triangle = side length * √3/2
        height = size * np.sqrt(3) / 2

        if self.size == "small":
            # For small triangles, increase relative size for better visibility
            height *= 1.2

        # Define triangle points
        points = [
            (x, y - height/2),  # top
            (x - half_size, y + height/2),  # bottom left
            (x + half_size, y + height/2),  # bottom right
        ]

        if self.rotation != 0:
            # Rotate points around the center
            rad = np.radians(self.rotation)
            cos_val, sin_val = np.cos(rad), np.sin(rad)
            points = [(cos_val * (px - x) - sin_val * (py - y) + x,
                       sin_val * (px - x) + cos_val * (py - y) + y) for px, py in points]

        if self.style == "solid":
            # Solid fill
            draw.polygon(points, fill=COLORS[self.color1])
        elif self.style == "half":
            # For half-color style, we create a custom masked polygon
            # First, find the bounding box of the triangle
            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)
            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)

            width = int(max_x - min_x) + 2  # Add padding
            height = int(max_y - min_y) + 2

            # Create a temporary mask
            temp_mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_mask)

            # Adjust points to local coordinates
            local_points = [(px - min_x, py - min_y) for px, py in points]

            # Draw the triangle in white
            temp_draw.polygon(local_points, fill=(255, 255, 255, 255))

            # Create a color mask
            color_mask = self._create_half_color_mask(
                width, height, COLORS[self.color1], COLORS[self.color2]
            )

            # Combine the masks
            result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            result.paste(color_mask, (0, 0), temp_mask)

            # Paste onto the main image
            draw._image.paste(result, (int(min_x), int(min_y)), result)
        elif self.style == "border":
            # Draw outline
            draw.polygon(points, outline=COLORS[self.color2], fill=COLORS[self.color1], width=3)

    def _draw_square(self, draw: ImageDraw.ImageDraw) -> None:
        """Draw a square on the ImageDraw object."""
        x, y = self.position
        size = self.size_px
        half_size = size // 2

        # Define corner points
        left, top = x - half_size, y - half_size
        right, bottom = x + half_size, y + half_size

        if self.rotation != 0:
            # For a rotated square, use polygon approach
            points = [(left, top), (right, top), (right, bottom), (left, bottom)]

            # Rotate points around the center
            rad = np.radians(self.rotation)
            cos_val, sin_val = np.cos(rad), np.sin(rad)
            points = [(cos_val * (px - x) - sin_val * (py - y) + x,
                       sin_val * (px - x) + cos_val * (py - y) + y) for px, py in points]

            if self.style == "solid":
                # Solid fill
                draw.polygon(points, fill=COLORS[self.color1])
            elif self.style == "half":
                # For half-color style, we use the same approach as triangle
                min_x = min(p[0] for p in points)
                max_x = max(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_y = max(p[1] for p in points)

                width = int(max_x - min_x) + 2
                height = int(max_y - min_y) + 2

                # Create masks
                temp_mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_mask)

                # Local coordinates
                local_points = [(px - min_x, py - min_y) for px, py in points]
                temp_draw.polygon(local_points, fill=(255, 255, 255, 255))

                # Color mask
                color_mask = self._create_half_color_mask(
                    width, height, COLORS[self.color1], COLORS[self.color2]
                )

                # Combine
                result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                result.paste(color_mask, (0, 0), temp_mask)

                # Paste on main image
                draw._image.paste(result, (int(min_x), int(min_y)), result)
            elif self.style == "border":
                draw.polygon(points, outline=COLORS[self.color2], fill=COLORS[self.color1], width=3)
        else:
            # No rotation, use rectangle directly
            if self.style == "solid":
                draw.rectangle((left, top, right, bottom), fill=COLORS[self.color1])
            elif self.style == "half":
                # Create a rectangle mask and apply the half coloring
                mask = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)

                # Draw a white rectangle for the mask
                mask_draw.rectangle((0, 0, size, size), fill=(255, 255, 255, 255))

                # Create a color mask
                color_mask = self._create_half_color_mask(
                    size, size, COLORS[self.color1], COLORS[self.color2]
                )

                # Combine
                result = Image.new('RGBA', (size, size), (0, 0, 0, 0))
                result.paste(color_mask, (0, 0), mask)

                # Paste on main image
                draw._image.paste(result, (left, top), result)
            elif self.style == "border":
                draw.rectangle((left, top, right, bottom), outline=COLORS[self.color2], fill=COLORS[self.color1], width=3)

    def _draw_circle(self, draw: ImageDraw.ImageDraw) -> None:
        """Draw a circle on the ImageDraw object."""
        x, y = self.position
        size = self.size_px
        left, top, right, bottom = self.bbox

        if self.style == "solid":
            draw.ellipse(self.bbox, fill=COLORS[self.color1])
        elif self.style == "half":
            # For half-color style, create a circle and apply the dual coloring
            mask = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask)

            # Draw a white circle for the mask
            mask_draw.ellipse((0, 0, size-1, size-1), fill=(255, 255, 255, 255))

            # Create color mask
            color_mask = self._create_half_color_mask(
                size, size, COLORS[self.color1], COLORS[self.color2]
            )

            # Combine
            result = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            result.paste(color_mask, (0, 0), mask)

            # Paste to main image
            image_x, image_y = left, top
            draw._image.paste(result, (image_x, image_y), result)
        elif self.style == "border":
            draw.ellipse(self.bbox, outline=COLORS[self.color2], fill=COLORS[self.color1], width=3)

    def to_dict(self) -> Dict:
        """Convert the shape object to a dictionary for annotation."""
        return {
            "shape_type": self.shape_type,
            "size": self.size,
            "color1": self.color1,
            "color2": self.color2,
            "style": self.style,
            "position": self.position,
            "rotation": self.rotation,
            "bbox": self.bbox
        }

def get_random_attributes():
    """Get random attributes for creating a shape from the config."""
    shape_type = np.random.choice(CONFIG["shape_types"])
    size = np.random.choice(CONFIG["sizes"])
    color1 = np.random.choice(list(CONFIG["colors"].keys()))
    style = np.random.choice(CONFIG["styles"])

    # For half and border styles, pick a second color
    color2 = None
    if style in ["half", "border"]:
        # Make sure color2 is different from color1
        available_colors = [c for c in CONFIG["colors"].keys() if c != color1]
        color2 = np.random.choice(available_colors)

    # Random rotation within the specified range for the shape
    min_rot, max_rot = ROTATION_RANGES[shape_type]
    rotation = np.random.uniform(min_rot, max_rot)

    return {
        "shape_type": shape_type,
        "size": size,
        "color1": color1,
        "color2": color2,
        "style": style,
        "rotation": rotation
    }
