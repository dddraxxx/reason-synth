# Project Management

## Current Version: v1.3

### Features
- Shape generation (triangle, square, circle)
- Shape attributes (size, color, style, rotation)
- Random grid layout in defined regions
- Minimum spacing between objects (1/4 of smaller object's bounding box)
- Proper bounding box calculation for rotated shapes
- Configurable grid sizes (2x2 up to 8x8)
- JSONL annotation format
- Visualization tools

### Project Guidelines

1. **Keep it simple**
   - Avoid unnecessary complexity
   - One function, one purpose
   - Clean interfaces between components

2. **Code organization**
   - Maintain clean directory structure
   - Remove unused/old code promptly
   - Minimize dependencies

3. **Development workflow**
   - Test changes with small sample sizes first
   - Commit working code only
   - Document significant changes

## Directory Structure

```
reason_synth/
├── src/                    # Core functionality
│   ├── shapes.py           # Shape classes and drawing
│   ├── scene.py            # Scene composition
│   ├── generator.py        # Dataset generation
│   └── object_config.json  # Shape attributes
├── samples/                # Sample outputs
├── better_spacing/         # Samples with improved object spacing
├── large_dataset/          # Large dataset with 200 samples
├── rotated_test/           # Test samples with rotated shape support
├── gallery.png             # Shape gallery
├── gallery_rotated.png     # Rotated shape gallery
├── generate_gallery.py     # Gallery generation
├── generate_samples.py     # Sample generation
├── visualize.py            # Visualization
├── README.md               # Documentation
└── MANAGEMENT.md           # This file
```

## Recent Changes

### v1.3 (Current)
- Added proper bounding box calculation for rotated shapes
- Improved minimum distance calculation to use bounding box diagonals
- Reduced maximum offsets for rotated shapes to prevent overlaps

### v1.2
- Added minimum spacing between objects (1/4 of smaller object's bounding box)
- Added support for configurable grid sizes (2x2 up to 8x8)
- Improved position validation to prevent objects from being too close

## Quick Reference

### Generate samples with varied grid sizes
```
python generate_samples.py --num-samples 4 --output-dir samples --min-grid 2 --max-grid 8
```

### Create gallery
```
python generate_gallery.py --output gallery.png --rotated-output gallery_rotated.png
```

### Visualize
```
python visualize.py --image samples/images/sample_0000.png --jsonl samples/annotations/dataset.jsonl --show-grid --show-bbox --show-region
```