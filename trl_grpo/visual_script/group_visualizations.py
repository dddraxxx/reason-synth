#!/usr/bin/env python3
import json
import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path
import textwrap
from collections import defaultdict
import datetime

def parse_bbox_from_answer(answer_text):
    """Extract bounding box coordinates from the model's answer."""
    # Try to extract standard format [x_min, y_min, x_max, y_max]
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(bbox_pattern, answer_text)
    if matches:
        return [[int(x) for x in match] for match in matches]

    # Try to extract JSON format with "bbox_2d" key
    json_pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(json_pattern, answer_text)
    if matches:
        return [[int(x) for x in match] for match in matches]

    # If no matches found, return empty list
    return []

def extract_think_content(model_output):
    """Extract the content within <think> tags."""
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, model_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No reasoning found"

def extract_answer_content(model_output):
    """Extract the content within <answer> tags."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, model_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No answer found"

def visualize_grouped_samples(group_key, samples, image_dir, output_dir, global_step_map={}):
    """Visualize multiple samples with the same reference and image as a single grouped image."""
    # Get image path from the first sample
    image_basename = os.path.basename(samples[0]['image_path'])
    image_path = os.path.join(image_dir, image_basename)
    print(f"Looking for image: {image_path}")

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        # Try to find the image in subdirectories
        for root, dirs, files in os.walk(image_dir):
            if image_basename in files:
                image_path = os.path.join(root, image_basename)
                print(f"Found image at: {image_path}")
                break
        else:
            print(f"Image not found anywhere in {image_dir} or its subdirectories")
            return False

    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate grid layout
    num_samples = len(samples)
    cols = min(3, num_samples)  # Maximum 3 columns
    rows = (num_samples + cols - 1) // cols  # Calculate needed rows

    # Create figure with subplots - adjust size for better spacing
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*8), constrained_layout=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Handle single subplot case
    if rows * cols == 1:
        axes = np.array([axes])

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Set a common title for the entire figure with improved styling
    ref_text = samples[0]['ref_text']
    fig.suptitle(f"Reference: {ref_text}", fontsize=16, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.8))

    # Process each sample
    for i, sample in enumerate(samples):
        if i >= len(axes):
            print(f"Warning: More samples than subplot spaces for {group_key}")
            break

        ax = axes[i]

        # Display image
        ax.imshow(np.array(img))

        # Extract ground truth bounding boxes
        gt_boxes = sample['ground_truth']

        # Extract predicted bounding boxes from model output
        pred_boxes = parse_bbox_from_answer(sample['model_output'])

        # Draw ground truth boxes in green
        for box in gt_boxes:
            if len(box) == 4:  # Ensure box has 4 coordinates
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth'
                )
                ax.add_patch(rect)

        # Draw predicted boxes in red
        for box in pred_boxes:
            if len(box) == 4:  # Ensure box has 4 coordinates
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='r', facecolor='none', label='Prediction'
                )
                ax.add_patch(rect)

        # Add legend (only once for each color)
        handles, labels = [], []
        if gt_boxes:
            handles.append(patches.Patch(color='g', label='Ground Truth'))
            labels.append('Ground Truth')
        if pred_boxes:
            handles.append(patches.Patch(color='r', label='Prediction'))
            labels.append('Prediction')

        if handles:
            ax.legend(handles=handles, labels=labels, loc='upper right')

        # Set title with sample ID and IoU reward - more compact and styled
        iou_reward = sample['rewards']['multi_bbox_iou_reward']
        ax.set_title(f"ID: {sample['sample_id']} | IoU: {iou_reward:.4f}",
                     fontsize=10, pad=5, fontweight='bold')

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add thinking process and answer below the subplot
        thinking = extract_think_content(sample['model_output'])
        answer = extract_answer_content(sample['model_output'])

        # Wrap text without truncation
        wrapped_thinking = textwrap.fill(thinking, width=80)
        wrapped_answer = textwrap.fill(answer, width=70)

        # Create a single text box with both thinking and answer sections
        # This ensures they don't overlap
        combined_text = f"THINKING:\n{wrapped_thinking}\n\nANSWER:\n{wrapped_answer}"

        # Create a text box with a gradient-like appearance
        combined_box = dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8', edgecolor='#cccccc', alpha=0.9)
        ax.text(0.5, -0.12, combined_text,
                transform=ax.transAxes, fontsize=8, ha='center', va='top',
                bbox=combined_box, wrap=True, family='sans-serif')

    # Hide any unused subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')

    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    # Create a safe filename from the group key
    safe_filename = re.sub(r'[^\w\-_]', '_', group_key)
    # Get the global step for this group
    global_step = global_step_map.get(group_key, 0)
    output_path = os.path.join(output_dir, f"step_{global_step:04d}_group_{safe_filename}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved grouped visualization to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Group and visualize GRPO results')
    parser.add_argument('--jsonl_file', type=str,
                        default='/mnt/localssd/reason-synth/trl_grpo/logs/qwen2_5_grpo_20250409_122101.jsonl',
                        help='Path to the JSONL file with training results')
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/localssd/reason-synth/rs1/images/',
                        help='Directory containing the images')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/localssd/reason-synth/trl_grpo/grouped_visualizations',
                        help='Directory to save grouped visualizations')
    parser.add_argument('--group_by', type=str, default='image_and_ref',
                        choices=['image_only', 'image_and_ref'],
                        help='How to group samples: by image only or by image and reference text')
    parser.add_argument('--sampling_freq', type=int, default=1,
                        help='Sample 1 out of N groups (default: 1, meaning keep all groups)')
    parser.add_argument('--samples_in_group', type=int, default=None,
                        help='Number of samples to show in each group visualization (default: None, meaning show all)')
    parser.add_argument('--group_period_start', type=float, default=0.0,
                        help='Start sampling groups from this point in the timeline (0-1, default: 0.0)')
    parser.add_argument('--group_period_end', type=float, default=1.0,
                        help='End sampling groups at this point in the timeline (0-1, default: 1.0)')
    parser.add_argument('--max_groups', type=int, default=5,
                        help='Maximum number of groups to visualize (default: 5, use -1 for all)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Warning: Image directory {args.image_dir} does not exist")
    else:
        print(f"Image directory: {args.image_dir}")

    # Check if JSONL file exists
    if not os.path.exists(args.jsonl_file):
        print(f"Error: JSONL file {args.jsonl_file} does not exist")
        return
    print(f"JSONL file: {args.jsonl_file}")

    # Read JSONL file
    samples = []
    try:
        with open(args.jsonl_file, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"Successfully loaded {len(samples)} samples from JSONL file")
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return

    # Improved grouping logic
    # First, extract all unique image paths and reference texts
    image_paths = set(os.path.basename(sample['image_path']) for sample in samples)
    ref_texts = set(sample['ref_text'] for sample in samples)

    print(f"Found {len(image_paths)} unique images and {len(ref_texts)} unique reference texts")

    # Create a mapping of samples by their image and reference text
    samples_by_image_ref = defaultdict(list)

    for sample in samples:
        image_key = os.path.basename(sample['image_path'])
        ref_key = sample['ref_text']

        # Add timestamp for sorting
        if 'timestamp' in sample:
            sample['_parsed_timestamp'] = sample['timestamp']
        else:
            # If no timestamp, use sample_id as fallback
            sample['_parsed_timestamp'] = str(sample['sample_id'])

        if args.group_by == 'image_only':
            key = image_key
        else:
            key = f"{image_key}_{ref_key}"

        samples_by_image_ref[key].append(sample)

    # Sort samples within each group by timestamp
    for key in samples_by_image_ref:
        samples_by_image_ref[key].sort(key=lambda x: x['_parsed_timestamp'])

    print(f"Grouped {len(samples)} samples into {len(samples_by_image_ref)} groups")

    # Apply sampling within each group if specified
    for key in samples_by_image_ref:
        group = samples_by_image_ref[key]
        if len(group) > 1:
            # Limit the number of samples per group if specified
            if args.samples_in_group is not None and len(samples_by_image_ref[key]) > args.samples_in_group:
                # Evenly sample the specified number of items
                indices = np.linspace(0, len(samples_by_image_ref[key]) - 1, args.samples_in_group, dtype=int)
                samples_by_image_ref[key] = [samples_by_image_ref[key][i] for i in indices]

                if args.debug:
                    print(f"Group {key}: Sampled {len(samples_by_image_ref[key])} out of {len(group)} samples")

    # Convert back to a regular dictionary for easier handling
    grouped_samples = dict(samples_by_image_ref)

    # Sort all groups by timestamp (using the first sample in each group)
    group_keys = list(grouped_samples.keys())
    group_keys.sort(key=lambda k: grouped_samples[k][0]['_parsed_timestamp'] if grouped_samples[k] else '')

    # Create a mapping of group keys to their global step (index ordered by time)
    global_step_map = {key: idx for idx, key in enumerate(group_keys)}

    # Apply group period sampling - only keep groups from the specified period range
    if args.group_period_start > 0 or args.group_period_end < 1.0:
        start_idx = max(0, int(len(group_keys) * args.group_period_start))
        end_idx = min(len(group_keys), int(len(group_keys) * args.group_period_end))

        print(f"Keeping groups {start_idx}-{end_idx} out of {len(group_keys)} (period: {args.group_period_start}-{args.group_period_end})")
        group_keys = group_keys[start_idx:end_idx]

    # Apply group sampling frequency (1 out of N groups)
    if args.sampling_freq > 1 and len(group_keys) > args.sampling_freq:
        sampled_keys = [group_keys[i] for i in range(0, len(group_keys), args.sampling_freq)]
        print(f"Sampling 1 out of {args.sampling_freq} groups: {len(sampled_keys)} out of {len(group_keys)}")
        group_keys = sampled_keys

    # Limit to max_groups if specified
    if args.max_groups > 0 and len(group_keys) > args.max_groups:
        print(f"Limiting visualization to {args.max_groups} groups out of {len(group_keys)}")
        group_keys = group_keys[:args.max_groups]

    # Visualize selected groups
    successful_visualizations = 0
    for group_key in group_keys:
        group_samples = grouped_samples[group_key]
        print(f"Visualizing group {group_key} with {len(group_samples)} samples")

        if visualize_grouped_samples(group_key, group_samples, args.image_dir, args.output_dir, global_step_map):
            successful_visualizations += 1

        # Print progress
        print(f"Processed {successful_visualizations}/{len(group_keys)} groups")

    print(f"Visualization complete. Successfully visualized {successful_visualizations}/{len(group_keys)} groups.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
