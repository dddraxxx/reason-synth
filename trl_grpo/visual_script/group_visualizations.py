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
import sys
import wandb
from typing import List, Dict, Tuple, Optional, Any

# Get the root directory (parent of the current file directory)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_bbox_from_answer(answer_text):
    """Extract bounding box coordinates from the model's answer."""
    # Try to extract standard format [x_min, y_min, x_max, y_max]
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(bbox_pattern, answer_text, re.DOTALL)
    if matches:
        return [[int(x) for x in match] for match in matches]

    # Try to extract JSON format with "bbox_2d" key
    json_pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches = re.findall(json_pattern, answer_text, re.DOTALL)
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
    return model_output

def extract_answer_content(model_output):
    """Extract the content within <answer> tags."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, model_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No answer found"

# --- W&B Helper Functions Start ---
def get_run_steps(run) -> Tuple[int, int]:
    """
    Get the start and end global steps for a run by scanning its history for 'train/global_step'.

    Args:
        run: W&B run object

    Returns:
        tuple: (min_step, max_step)
    """
    min_step = float('inf')
    max_step = float('-inf')
    found_steps = False
    step_key = 'train/global_step'  # Target the specific step key

    try:
        # Scan history for the 'train/global_step' column
        print(f"    Scanning history for key: '{step_key}'...")
        history = run.scan_history(keys=[step_key])

        for row in history:
            if step_key in row and row[step_key] is not None:  # Check for key and non-null value
                step = row[step_key]
                # Ensure step is numeric before comparison
                if isinstance(step, (int, float)):
                    min_step = min(min_step, step)
                    max_step = max(max_step, step)
                    found_steps = True

        if found_steps:
            # Ensure we return integers
            return int(min_step), int(max_step)
        else:
            print(f"    Warning: Key '{step_key}' not found or had no numeric values in history for run {run.id}. Falling back.")
            # Fallback if 'train/global_step' not found or empty
            start_step = run.starting_step if hasattr(run, 'starting_step') else 0
            end_step = run.step if hasattr(run, 'step') else 0  # Use default _step as fallback
            end_step = max(start_step, end_step)
            return start_step, end_step

    except Exception as e:
        print(f"    Error scanning history for '{step_key}' in run {run.id}: {e}. Falling back.")
        # Fallback to basic attributes on error
        start_step = run.starting_step if hasattr(run, 'starting_step') else 0
        end_step = run.step if hasattr(run, 'step') else 0
        end_step = max(start_step, end_step)
        return start_step, end_step

def get_matching_runs_and_files(api, run_path: Optional[str] = None, run_name: Optional[str] = None,
                               jsonl_pattern: Optional[str] = None) -> List[Tuple[Any, List[Any]]]:
    """
    Find all runs and their matching JSONL files.

    Args:
        api: wandb.Api instance
        run_path: Direct path to a run (entity/project/run_id), can include wildcards like 'entity/project/*'
        run_name: Run name to filter by. Should be in entity/project/run_name format.
        jsonl_pattern: Regex pattern to match specific jsonl files

    Returns:
        List of (run, matching_files) pairs
    """
    matching_runs_files = []

    if run_path:
        try:
            # Check if run_path contains wildcards
            if '*' in run_path:
                # Split path and extract components
                parts = run_path.split('/')
                if len(parts) < 2:
                    raise ValueError("Invalid run path format. Should be 'entity/project/run_id' (run_id can be a wildcard)")

                entity = parts[0]
                project = parts[1]
                run_id_pattern = parts[2] if len(parts) > 2 else '*'

                print(f"Searching for runs in {entity}/{project} with ID pattern: {run_id_pattern}")

                # Convert wildcard to a regex pattern
                run_id_regex = run_id_pattern.replace('*', '.*')

                # Get all runs in the project
                project_runs = api.runs(f"{entity}/{project}")
                if not project_runs:
                    raise FileNotFoundError(f"No runs found in project {entity}/{project}")

                # Filter runs based on the pattern
                matched_runs = []
                for run in project_runs:
                    if re.match(f"^{run_id_regex}$", run.id):
                        matched_runs.append(run)

                if not matched_runs:
                    raise FileNotFoundError(f"No runs match the pattern '{run_id_pattern}' in {entity}/{project}")

                print(f"Found {len(matched_runs)} matching runs")

                # Process each matched run
                for run in matched_runs:
                    files = list(run.files())
                    jsonl_files = [f for f in files if f.name.endswith('.jsonl')]

                    if not jsonl_files:
                        print(f"No JSONL files found in run {run.id}")
                        continue

                    # Apply pattern filter if provided
                    if jsonl_pattern:
                        pattern = re.compile(jsonl_pattern)
                        matching_files = [f for f in jsonl_files if pattern.search(f.name)]
                        if matching_files:
                            matching_runs_files.append((run, matching_files))
                            print(f"Found {len(matching_files)} files in run {run.id} matching pattern '{jsonl_pattern}'")
                        else:
                            print(f"No JSONL files in run {run.id} match pattern '{jsonl_pattern}'")
                    else:
                        matching_runs_files.append((run, jsonl_files))
            else:
                # Direct path to a specific run
                print(f"Connecting to wandb run: {run_path}")
                run = api.run(run_path)

                # Get all JSONL files for this run
                files = list(run.files())
                jsonl_files = [f for f in files if f.name.endswith('.jsonl')]

                if not jsonl_files:
                    print(f"No JSONL files found in run {run.id}. Available files: {[f.name for f in files]}")
                    return []

                # Apply pattern filter if provided
                if jsonl_pattern:
                    pattern = re.compile(jsonl_pattern)
                    matching_files = [f for f in jsonl_files if pattern.search(f.name)]
                    if matching_files:
                        matching_runs_files.append((run, matching_files))
                        print(f"Found {len(matching_files)} files in run {run.id} matching pattern '{jsonl_pattern}'")
                    else:
                        print(f"No JSONL files in run {run.id} match pattern '{jsonl_pattern}'")
                else:
                    matching_runs_files.append((run, jsonl_files))

        except ValueError as e:
            raise ValueError(f"Error with run_path: {e}")

    elif run_name:
        # run_name should be in entity/project/run_name format
        try:
            # Parse entity and project from run_name
            parts = run_name.split('/')
            if len(parts) == 3:
                entity_val, project_val, name_val = parts
            else:
                raise ValueError("run_name must be in format 'entity/project/run_name'")

            # Query by run name approach
            print(f"Connecting to wandb project: {entity_val}/{project_val}")
            print(f"Searching for runs with name: {name_val}")

            # Filter runs by name
            runs = api.runs(f"{entity_val}/{project_val}", filters={"display_name": name_val})

            if not runs:
                raise FileNotFoundError(f"No runs found with name {name_val} in {entity_val}/{project_val}")

            print(f"Found {len(runs)} runs with name: {name_val}")

            # For each run, find matching JSONL files
            for run in runs:
                files = list(run.files())
                jsonl_files = [f for f in files if f.name.endswith('.jsonl')]

                if not jsonl_files:
                    print(f"No JSONL files found in run {run.id}")
                    continue

                # Apply pattern filter if provided
                if jsonl_pattern:
                    pattern = re.compile(jsonl_pattern)
                    matching_files = [f for f in jsonl_files if pattern.search(f.name)]
                    if matching_files:
                        matching_runs_files.append((run, matching_files))
                        print(f"Found {len(matching_files)} files in run {run.id} matching pattern '{jsonl_pattern}'")
                    else:
                        print(f"No JSONL files in run {run.id} match pattern '{jsonl_pattern}'")
                else:
                    matching_runs_files.append((run, jsonl_files))
        except ValueError as e:
            raise ValueError(f"Error with run_name: {e}")

    else:
        raise ValueError("Either run_path or run_name must be provided")

    return matching_runs_files

def select_best_run_by_steps(run_file_pairs: List[Tuple[Any, List[Any]]]) -> Tuple[Any, List[Any], Optional[Tuple[int, int]]]:
    """
    Let the user select the best run from matching runs or automatically select if only one match.

    Args:
        run_file_pairs: List of (run, matching_files) pairs

    Returns:
        Tuple[run, matching_files, run_steps]: Selected run, its matching files, and run step range
    """
    if not run_file_pairs:
        raise FileNotFoundError("No runs with matching JSONL files found")

    # If only one run, use it automatically
    if len(run_file_pairs) == 1:
        best_run, best_files = run_file_pairs[0]
        best_run_steps = get_run_steps(best_run)
        print(f"Selected run: {best_run.name} ({best_run.id}) with step range {best_run_steps[0]}-{best_run_steps[1]}")
        return best_run, best_files, best_run_steps

    # Multiple runs found - let user choose
    print(f"\nFound {len(run_file_pairs)} matching runs:")
    print("-" * 100)
    print(f"{'#':<3} {'Run Name':<30} {'Run ID':<9} {'Created':<16} {'Step Range':<15} {'Files':<10} {'Description'}")
    print("-" * 100)

    # Display runs with details
    for i, (run, files) in enumerate(run_file_pairs):
        step_range = get_run_steps(run)
        step_str = f"{step_range[0]}-{step_range[1]}"

        # Format created time to readable format
        created_time = run.created_at if hasattr(run, 'created_at') else "N/A"

        # Truncate and clean run name if too long
        name_display = run.name[:27] + "..." if len(run.name) > 30 else run.name

        # Get file count and names
        file_str = f"{len(files)} files"

        # Get and truncate description
        description = run.description if hasattr(run, 'description') and run.description else "No description"
        desc_display = description[:40] + "..." if len(description) > 40 else description

        # Print row
        print(f"{i+1:<3} {name_display:<30} {run.id:<9} {created_time:<16} {step_str:<15} {file_str:<10} {desc_display}")

    print(f"Enter the number of the run you want to use, or:")
    print(f"- Press d + number to see details for a specific run (e.g., d2)")
    print(f"- Press q to quit")

    # Let user choose
    while True:
        try:
            selection = input(f"Your selection (1-{len(run_file_pairs)}, d#, or q): ").strip().lower()

            # Check for quit command
            if selection == 'q':
                print("Exiting...")
                sys.exit(0)

            # Check for details command
            if selection.startswith('d') and len(selection) > 1:
                try:
                    detail_idx = int(selection[1:]) - 1
                    if 0 <= detail_idx < len(run_file_pairs):
                        run_to_show, files_to_show = run_file_pairs[detail_idx]
                        steps = get_run_steps(run_to_show)
                        print(f"\n--- Detailed information for run #{detail_idx+1} ---")
                        print(f"Name: {run_to_show.name}")
                        print(f"ID: {run_to_show.id}")
                        print(f"Project: {run_to_show.project}")
                        print(f"Entity: {run_to_show.entity}")
                        print(f"Created: {run_to_show.created_at if hasattr(run_to_show, 'created_at') else 'N/A'}")
                        print(f"Step Range: {steps[0]}-{steps[1]}")
                        print(f"Description: {run_to_show.description if hasattr(run_to_show, 'description') and run_to_show.description else 'No description'}")
                        print(f"Tags: {', '.join(run_to_show.tags) if hasattr(run_to_show, 'tags') and run_to_show.tags else 'No tags'}")
                        print(f"JSONL Files ({len(files_to_show)}):")
                        for j, file in enumerate(files_to_show):
                            print(f"  {j+1}. {file.name} ({file.size/1024/1024:.2f}MB)")
                        print("---")
                        continue
                    else:
                        print(f"Invalid run number. Please enter d1-d{len(run_file_pairs)}")
                        continue
                except ValueError:
                    print(f"Invalid detail command. Use d1-d{len(run_file_pairs)}")
                    continue

            # Regular number selection
            idx = int(selection) - 1
            if 0 <= idx < len(run_file_pairs):
                selected_run, selected_files = run_file_pairs[idx]
                run_steps = get_run_steps(selected_run)
                print(f"Selected run: {selected_run.name} ({selected_run.id}) with step range {run_steps[0]}-{run_steps[1]}")
                return selected_run, selected_files, run_steps
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(run_file_pairs)}")
        except ValueError:
            print("Please enter a valid selection")

def download_best_file(run: Any, files: List[Any], download_dir: Path) -> str:
    """
    Download the best file from the selected run.

    Args:
        run: W&B run object
        files: List of file objects
        download_dir: Directory to download to

    Returns:
        str: Path to downloaded file
    """
    # For now, just use the first matching file
    selected_file = files[0]
    local_path = download_dir / selected_file.name

    # Download the file
    print(f"Downloading {selected_file.name} from run {run.id}...")
    selected_file.download(root=str(download_dir), replace=True)

    print(f"Downloaded to {local_path}")
    return str(local_path)

def download_from_wandb(run_path: Optional[str] = None, run_name: Optional[str] = None,
                       jsonl_pattern: Optional[str] = None) -> Tuple[str, Optional[Tuple[int, int]], Optional[Dict]]:
    """
    Download the debug log file from wandb run.

    Args:
        run_path (Optional[str]): Path to the wandb run (e.g., 'entity/project/run_id')
        run_name (Optional[str]): Run name to filter by. Must be in format 'entity/project/run_name'
        jsonl_pattern (Optional[str]): Regex pattern to match specific jsonl files.

    Returns:
        Tuple[str, Optional[Tuple[int, int]], Optional[Dict]]: (Path to the downloaded file, Run's full step range, Run info)
    """
    # Create download directory
    download_dir = Path("./tmp/wandb_downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Initialize API
    api = wandb.Api()

    try:
        # 1. Find all runs with matching JSONL files
        matching_runs_files = get_matching_runs_and_files(
            api=api,
            run_path=run_path,
            run_name=run_name,
            jsonl_pattern=jsonl_pattern
        )

        if not matching_runs_files:
            if jsonl_pattern:
                raise FileNotFoundError(f"No runs found with JSONL files matching pattern '{jsonl_pattern}'")
            else:
                raise FileNotFoundError("No runs found with JSONL files")

        # 2. Select the best run
        best_run, best_files, run_steps = select_best_run_by_steps(
            run_file_pairs=matching_runs_files
        )

        # Extract runtime information
        runtime_str = "unknown"
        if hasattr(best_run, 'created_at'):
            created_time = best_run.created_at
            runtime_str = created_time

        # Extract run info
        run_info = {
            'name': best_run.name,
            'id': best_run.id,
            'runtime': runtime_str
        }

        # 3. Download the best file
        local_path = download_best_file(
            run=best_run,
            files=best_files,
            download_dir=download_dir
        )

        return local_path, run_steps, run_info

    except Exception as e:
        print(f"Error downloading from wandb: {e}")
        raise

# --- W&B Helper Functions End ---

def visualize_grouped_samples(group_key, samples, image_dir, output_dir, global_step_map={}, run_info=None):
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
    # Create a run-specific subdirectory if run_info is provided
    if run_info and 'name' in run_info and 'id' in run_info:
        # Include runtime if available
        if 'runtime' in run_info and run_info['runtime']:
            run_subdir = f"{run_info['name']}_{run_info['runtime']}_{run_info['id']}"
        else:
            run_subdir = f"{run_info['name']}_{run_info['id']}"
        # Clean the run subdirectory name to be filesystem-friendly
        run_subdir = re.sub(r'[^\w\-_]', '_', run_subdir)
        # Create the full output path including the run subdirectory
        full_output_dir = os.path.join(output_dir, run_subdir)
    else:
        full_output_dir = output_dir

    os.makedirs(full_output_dir, exist_ok=True)

    # Create a safe filename from the group key
    safe_filename = re.sub(r'[^\w\-_]', '_', group_key)
    # Get the global step for this group
    global_step = global_step_map.get(group_key, 0)
    output_path = os.path.join(full_output_dir, f"step_{global_step:04d}_group_{safe_filename}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved grouped visualization to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Group and visualize GRPO results')
    # Input source arguments
    input_group = parser.add_argument_group('Input Source (Provide ONE)')
    # input_source = input_group.add_mutually_exclusive_group(required=True)
    input_source = input_group
    input_source.add_argument('--input', '-i', type=str,
                        help='Path to the input JSONL file')
    input_source.add_argument('--wandb-run', '-wr', type=str, default='*',
                        help='Wandb run path (e.g., entity/project/run_id). Supports wildcards for run_id (e.g., entity/project/abc* or entity/project/*)')
    input_source.add_argument('--run-name', type=str,
                        help='Filter runs by name. Must be in format entity/project/run_name')

    # W&B related arguments
    wandb_group = parser.add_argument_group('W&B Options')
    wandb_group.add_argument('--jsonl-pattern', type=str, default='jsonl',
                        help='Regex pattern to match specific jsonl files in the run (optional)')
    wandb_group.add_argument('--entity', type=str, default='iccv25_cost',
                        help='Wandb entity (default: iccv25_cost)')
    wandb_group.add_argument('--project', type=str, default='reason-synth',
                        help='Wandb project (default: reason-synth)')

    # Existing arguments
    parser.add_argument('--image_dir', type=str,
                        default='../rs2/extreme_simple_dfs/images/',
                        help='Directory containing the images (relative to root_dir if no leading /)')
    parser.add_argument('--output_dir', type=str,
                        default='./grouped_visualizations',
                        help='Directory to save grouped visualizations (relative to root_dir if no leading /)')
    parser.add_argument('--group_by', type=str, default='image_and_ref',
                        choices=['image_only', 'image_and_ref'],
                        help='How to group samples: by image only or by image and reference text')
    parser.add_argument('--sampling_freq', '-sf', type=int, default=1,
                        help='Sample 1 out of N groups (default: 1, meaning keep all groups)')
    parser.add_argument('--samples_in_group', '-sig', type=int, default=None,
                        help='Number of samples to show in each group visualization (default: None, meaning show all)')
    parser.add_argument('--group_period_start', '-gps', type=float, default=0.9,
                        help='Start sampling groups from this point in the timeline (0-1, default: 0.0)')
    parser.add_argument('--group_period_end', '-gpe', type=float, default=1.0,
                        help='End sampling groups at this point in the timeline (0-1, default: 1.0)')
    parser.add_argument('--max_groups', '-mg', type=int, default=None,
                        help='Maximum number of groups to visualize (default: None, meaning visualize all selected groups)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')

    args = parser.parse_args()

    # --- Input Handling ---
    input_path = None
    effective_run_range = None
    run_info = None  # Store run name and ID information

    # Process relative paths
    if args.input and not args.input.startswith('/'):
        args.input = os.path.join(root_dir, args.input)

    if not args.image_dir.startswith('/'):
        args.image_dir = os.path.join(root_dir, args.image_dir)

    if not args.output_dir.startswith('/'):
        args.output_dir = os.path.join(root_dir, args.output_dir)

    print(f"Root directory: {root_dir}")
    print(f"Resolved paths:")
    print(f"  Input: {args.input if args.input else 'Not provided'}")
    print(f"  Image directory: {args.image_dir}")
    print(f"  Output directory: {args.output_dir}")

    # Determine input_path based on provided arguments
    if args.wandb_run or args.run_name:
        print("Attempting to download JSONL from Weights & Biases...")
        try:
            # If run_path or run_name doesn't contain '/', use entity and project
            run_path = args.wandb_run
            if run_path and '/' not in run_path:
                run_path = f"{args.entity}/{args.project}/{run_path}"

            run_name = args.run_name
            if run_name and '/' not in run_name:
                run_name = f"{args.entity}/{args.project}/{run_name}"

            # Get the information from W&B
            input_path, effective_run_range, run_info = download_from_wandb(
                run_path=run_path,
                run_name=run_name,
                jsonl_pattern=args.jsonl_pattern
            )

            print("--- W&B Download Summary ---")
            print(f"Downloaded file path: {input_path}")
            if effective_run_range:
                print(f"Full run step range: {effective_run_range[0]}-{effective_run_range[1]}")
            if run_info:
                print(f"Run name: {run_info['name']}, ID: {run_info['id']}")
            print("--------------------------")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.input:
        input_path = args.input
        print(f"Using local input file: {input_path}")
        # For local files, use current datetime as subdirectory name
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_info = {
            'name': 'local',
            'id': current_time
        }
        print(f"Using timestamp as ID: {current_time}")
    else:
        # This case should not be reachable due to mutually_exclusive_group(required=True)
        print("Error: No input source specified (--input, --wandb-run, or --run-name).")
        sys.exit(1)

    # --- End Input Handling ---

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Check if image directory exists
    image_dir_path = Path(args.image_dir)
    if not image_dir_path.exists():
        print(f"Warning: Image directory {image_dir_path} does not exist")
    else:
        print(f"Image directory: {image_dir_path}")

    # Check if JSONL file exists
    input_file_path = Path(input_path)
    if not input_file_path.exists():
        print(f"Error: Input file {input_file_path} does not exist (after potential download)")
        sys.exit(1)
    print(f"Using JSONL file: {input_file_path}")

    # Read JSONL file
    samples = []
    try:
        with open(input_file_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"Successfully loaded {len(samples)} samples from JSONL file")
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        sys.exit(1)

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
    if args.max_groups is not None and len(group_keys) > args.max_groups:
        print(f"Limiting visualization to {args.max_groups} groups out of {len(group_keys)}")
        group_keys = group_keys[:args.max_groups]

    # Visualize selected groups
    successful_visualizations = 0
    for group_key in group_keys:
        group_samples = grouped_samples[group_key]
        print(f"Visualizing group {group_key} with {len(group_samples)} samples")

        if visualize_grouped_samples(group_key, group_samples, args.image_dir, args.output_dir, global_step_map, run_info):
            successful_visualizations += 1

        # Print progress
        print(f"Processed {successful_visualizations}/{len(group_keys)} groups")

    # Create a friendly display of where output is stored
    if run_info:
        # Include runtime if available
        if 'runtime' in run_info and run_info['runtime']:
            run_subdir = f"{run_info['name']}_{run_info['runtime']}_{run_info['id']}"
        else:
            run_subdir = f"{run_info['name']}_{run_info['id']}"
        run_subdir = re.sub(r'[^\w\-_]', '_', run_subdir)
        full_output_path = os.path.join(args.output_dir, run_subdir)
    else:
        full_output_path = args.output_dir

    print(f"Visualization complete. Successfully visualized {successful_visualizations}/{len(group_keys)} groups.")
    print(f"Results saved to {full_output_path}")

if __name__ == "__main__":
    main()
