# %%capture
import logging
import numpy as np
import os
os.environ["WANDB_PROJECT"] = "reason-synth"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig
import json
from datetime import datetime

#%%
from datasets import load_dataset
dataset = load_dataset("dddraxxx/reason_synth", split = "train")

#%%
dataset[0]["prompt"]

QUESTION_TEMPLATE = """
Please find the corresponding bounding box for {referring_expression}.

First output your reasoning process within <think> </think> tags and then output the answer, as JSON, within <answer> </answer> tags. The JSON should include the bounding box coordinates [x_min, y_min, x_max, y_max].
If no matching shape is found, return "not exist" in answer. If multiple shapes match, output all bounding boxes in answer.
"""
# The image contains shapes (triangles, squares, or circles) of at most two sizes (small or large), colors (red, blue, green, yellow, purple, orange), and at most three styles (solid-filled, two-color split, or outlined).

conv_prompt = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": QUESTION_TEMPLATE},
        ],
    },
]

import copy
def make_conversation_image(example):
    prompt = copy.deepcopy(conv_prompt)
    prompt[-1]["content"][-1]["text"] = QUESTION_TEMPLATE.format(referring_expression=example["referring_expression"])
    absolute_answer = [m['bbox'] for m in json.loads(example["matching_objects"])]
    # normalized_answer = [m['bbox'] for m in json.loads(example["matching_objects"])]
    # image_width, image_height = example["image"].size
    # absolute_answer = []
    # for box in normalized_answer:
    #     x_min = int(box[0]/1000 * image_width)
    #     y_min = int(box[1]/1000 * image_height)
    #     x_max = int(box[2]/1000 * image_width)
    #     y_max = int(box[3]/1000 * image_height)
    #     absolute_answer.append([x_min, y_min, x_max, y_max])

    return {
        "prompt": prompt,
        'ref_text': example["referring_expression"],
        'solution': absolute_answer,
    }

#%%
converted_dataset = dataset.map(make_conversation_image, num_proc=10)

#%%
converted_dataset[0]

#%% Different reward functions
def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

def iou_reward(completions, solution, iou_reward_type='cont', **kwargs):
    """Extract bounding boxes from completions and compute IoU rewards against solution.

    Args:
        completions: List of completions, where each completion is a list of messages
        solution: List of ground truth bounding boxes, where each element is a list of bbox coordinates [x_min, y_min, x_max, y_max]
                  Can be empty list (no match) or contain multiple bboxes (multiple matches)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    iou_reward_type = iou_reward_type

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    this_iou = iou(bbox, sol)
                    if iou_reward_type == 'binary':
                        if this_iou > 0.5:
                            reward = 1.0
                    else:  # continuous
                        reward = this_iou
        except Exception:
            pass
        rewards.append(reward)
    return rewards

from scipy.optimize import linear_sum_assignment
def multi_bbox_iou_reward(completions, solution, iou_threshold_low=0.1, iou_threshold_high=0.9, **kwargs):
    """Extract multiple bounding boxes from completions and compute IoU rewards against solution.
    Uses bipartite matching to find optimal assignment between predicted and ground truth boxes.

    Args:
        completions: List of completions, where each completion is a list of messages
        solution: List of lists of ground truth bounding boxes, where each inner list is a list of bbox coordinates
                 [x_min, y_min, x_max, y_max] for each sample
        iou_threshold_low: IoU values below this threshold will be set to 0
        iou_threshold_high: IoU values above this threshold will be set to 1
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    matching_info = []

    for content, gt_boxes in zip(contents, solution):
        reward = 0.0
        match_info = {
            "pred_boxes": [],
            "gt_boxes": gt_boxes,
            "matches": [],
            "match_ious": [],
            "reward": 0.0
        }

        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if not content_answer_match:
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            content_answer = content_answer_match.group(1).strip()

            # Check for "no match" response
            if "not exist" in content_answer.lower():
                # If ground truth also has no boxes, this is correct
                if len(gt_boxes) == 0:
                    reward = 1.0
                match_info["pred_boxes"] = "not exist"
                match_info["reward"] = reward
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            # Find all bounding boxes in the answer - now using re.DOTALL
            all_bbox_matches = re.findall(bbox_pattern, content_answer, re.DOTALL)
            if not all_bbox_matches:
                # No boxes found in prediction
                if len(gt_boxes) == 0:
                    reward = 1.0  # Correctly predicted no boxes
                match_info["reward"] = reward
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            # Convert matches to lists of coordinates
            pred_boxes = []
            for match in all_bbox_matches:
                try:
                    box = [int(float(match[0])), int(float(match[1])), int(float(match[2])), int(float(match[3]))]
                    pred_boxes.append(box)
                except ValueError:
                    continue  # Skip malformed boxes

            match_info["pred_boxes"] = pred_boxes

            # Handle special cases
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                reward = 1.0  # Both prediction and ground truth have no boxes
                match_info["reward"] = reward
                rewards.append(reward)
                matching_info.append(match_info)
                continue

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                # One has boxes, the other doesn't
                match_info["reward"] = reward
                rewards.append(reward)  # reward stays 0
                matching_info.append(match_info)
                continue

            # Create IoU matrix for bipartite matching
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
            raw_iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))  # Store raw IoU values before thresholding

            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    raw_iou = iou(pred_box, gt_box)
                    raw_iou_matrix[i, j] = raw_iou

                    # Apply IoU thresholding
                    this_iou = raw_iou
                    if this_iou < iou_threshold_low:
                        this_iou = 0.0
                    elif this_iou > iou_threshold_high:
                        this_iou = 1.0
                    iou_matrix[i, j] = this_iou

            # Convert to cost matrix (1 - IoU)
            cost_matrix = 1 - iou_matrix

            # Find optimal assignment using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_ious = [iou_matrix[i, j] for i, j in zip(row_indices, col_indices)]
            raw_matched_ious = [raw_iou_matrix[i, j] for i, j in zip(row_indices, col_indices)]

            # Store matching information
            match_info["matches"] = list(zip(row_indices.tolist(), col_indices.tolist()))
            match_info["match_ious"] = raw_matched_ious

            # Calculate reward - only continuous mode now
            if matched_ious:
                # Just use the sum of all IoUs
                reward = sum(matched_ious)

            match_info["reward"] = reward

        except Exception as e:
            print(f"Error in multi_bbox_iou_reward: {e}")
            pass

        rewards.append(reward)
        matching_info.append(match_info)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

save_file = False
def log_reward(completions, solution, **kwargs):
    """Combined reward function that calculates all rewards and handles logging."""
    global save_file
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Use provided rewards
    if "rewards" not in kwargs:
        return [0 for _ in range(len(completions))]
    rewards_dict = kwargs["rewards"]

    # Log results if in debug mode
    log_path = os.getenv("LOG_PATH")
    contents = [completion[0]["content"] for completion in completions]

    for idx, content in enumerate(contents):
        # Create JSON log entry
        log_entry = {
            "timestamp": current_time,
            "sample_id": idx,
            "prompt": kwargs.get('prompts', [['None']])[0][-1],
            "model_output": content,
            "ground_truth": solution[idx],
            "rewards": {name: float(values[idx]) for name, values in rewards_dict.items()},
            "image_path": kwargs.get('image_path', ['None'])[idx] if isinstance(kwargs.get('image_path', ['None']), list) else kwargs.get('image_path', 'None'),
            "ref_text": kwargs.get('ref_text', '')[idx]
        }

        # Write JSON entry
        if not os.path.exists(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if not save_file and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            save_file = True
            import wandb
            if wandb.run is not None:
                wandb.save(log_path, policy="live")
    return [0 for _ in range(len(completions))]

reward_funcs_registry = {
    # "accuracy": iou_reward,
    "multi_box_accuracy": multi_bbox_iou_reward,
    "format": format_reward,
    "log": log_reward,
}

#%%

from trl.trainer import GRPOConfig, GRPOTrainer, ModelConfig
from vllm_grpo_trainer import VisionGRPOVLLMTrainer

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = "Qwen/Qwen2.5-VL-3B-Instruct"
model_config = dict(
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=True,
)

# peft_config = get_peft_config(model_config)
peft_config = None

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 2e-5,
    # weight_decay = 0.1,
    warmup_ratio = 0.1,
    # lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 7, # Decrease if out of memory
    vllm_device=os.getenv("VLLM_DEVICE", "auto"),
    max_prompt_length = 1424,
    max_completion_length = 700,
    num_train_epochs = 5, # Set to 1 for a full training run
    # max_steps = 450,
    # save_steps = 450,
    # max_grad_norm = 0.5, # need to be the same as the gradient clipping in zero3.json
    gradient_checkpointing=True,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
    deepspeed="zero3.json",
)
training_args.model_init_kwargs = model_config

#%%
trainer = VisionGRPOVLLMTrainer(
    model = model,
    reward_funcs = list(reward_funcs_registry.values()),
    train_dataset = converted_dataset,
    args = training_args,
    peft_config = peft_config,
)
#%%
trainer_stats = trainer.train()
