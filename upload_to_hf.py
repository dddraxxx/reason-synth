#!/usr/bin/env python
import os
import json
import argparse
from pathlib import Path
from PIL import Image
import io
import numpy as np
from datasets import Dataset, Image as DsImage, Features, Value, Sequence

def load_jsonl(file_path):
    """Load jsonl file into a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

sample_prompts = """The image contains shapes (triangles, squares, or circles) of at most two sizes (small or large), colors (red, blue, green, yellow, purple, orange), and at most three styles (solid-filled, two-color split, or outlined).

Please find the corresponding bounding box for {referring_expression}.
Output your reasoning process within <think> </think> tags and the answer, as JSON, within <answer> </answer> tags. The JSON should include the bounding box coordinates [x_min, y_min, x_max, y_max].
If no matching shape is found, return "no match" in answer. If multiple shapes match, output all bounding boxes in answer."""

def process_item(item, base_dir):
    """Process a single item and return a dictionary."""
    # Check if image exists
    image_path = base_dir / item["image_path"]
    if not image_path.exists():
        print(f"Warning: Image {image_path} not found, skipping entry")
        return None

    # Load the image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Create and return dictionary for single item
    return {
        "image": image,
        "prompt": sample_prompts.format(referring_expression=item["referring_expression"])
    }

"""
python upload_to_hf.py -r rs2/extreme_simple_dfs -d dddraxxx/reason_synth_extreme_simple_dfs
"""
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Upload the Reason Synth dataset to Hugging Face")
    parser.add_argument("--root-dir", '-r', type=str, default="",
                        help="Root directory of the dataset (default: current directory)")
    parser.add_argument("--sample-limit", '-l', type=int, default=None,
                        help="Limit the number of samples to process (default: no limit)")
    parser.add_argument("--dataset-name", '-d', type=str, default="dddraxxx/reason_synth",
                        help="Name of the dataset on HuggingFace (default: dddraxxx/reason_synth)")
    parser.add_argument("--batch-size", '-b', type=int, default=1000,
                        help="Number of items to process in each batch (default: 1000)")
    args = parser.parse_args()
    if args.root_dir != "":
        base_dir = Path(args.root_dir)
    else:
        base_dir = Path(__file__).parent
    annotations_dir = base_dir / "annotations"

    # Load referring expressions data
    ref_exp_path = annotations_dir / "referring_expressions.jsonl"
    print(f"Loading referring expressions from {ref_exp_path}")
    ref_exps = load_jsonl(ref_exp_path)

    # Limit samples if requested
    if args.sample_limit is not None:
        ref_exps = ref_exps[:args.sample_limit]
        print(f"Limited to {args.sample_limit} samples")

    # Define processing function with required arguments
    for r in ref_exps:
        r['target_requirements'] = json.dumps(r['target_requirements'])
        r['matching_objects'] = json.dumps(r['matching_objects'])

    initial_dataset = Dataset.from_list(ref_exps)
    # Process items in parallel using map with 16 processes
    print(f"Processing {len(ref_exps)} items with 16 processes")
    full_dataset = initial_dataset.map(
        process_item,
        fn_kwargs={"base_dir": base_dir},
        num_proc=16,
    )

    # Display dataset info
    print("\nFull dataset info:")
    print(full_dataset)
    print(f"\nTotal samples: {len(full_dataset)}")

    # Display sample entry
    print("\nSample entry:")
    sample = full_dataset[0]
    for key, value in sample.items():
        if key != "image":  # Skip displaying image data
            print(f"  - {key}: {value}")

    # Upload the full dataset to the hub
    try:
        print(f"\nUploading full dataset to {args.dataset_name}")
        full_dataset.push_to_hub(
            args.dataset_name,
        )
        print(f"Full dataset successfully uploaded to {args.dataset_name}")

    except Exception as e:
        print(f"Error uploading dataset: {e}")
if __name__ == "__main__":
    main()