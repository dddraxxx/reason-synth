# %%capture
import logging
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

The image contains shapes (triangles, squares, or circles) of at most two sizes (small or large), colors (red, blue, green, yellow, purple, orange), and at most three styles (solid-filled, two-color split, or outlined).
First output your reasoning process within <think> </think> tags and then output the answer, as JSON, within <answer> </answer> tags. The JSON should include the bounding box coordinates [x_min, y_min, x_max, y_max].
If no matching shape is found, return "no match" in answer. If multiple shapes match, output all bounding boxes in answer.
"""

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
    gt_answer = [m['bbox'] for m in json.loads(example["matching_objects"])]
    return {
        "prompt": prompt,
        'ref_text': example["referring_expression"],
        'solution': gt_answer,
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

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def get_timestamp_log_path():
    """
    Get a log path with an index appended if the file already exists.
    Returns the final log path to use.
    """
    base_log_path = os.getenv("LOG_PATH")
    if not base_log_path:
        return None

    # Split the path into name and extension
    base_name = os.path.splitext(base_log_path)[0]
    ext = os.path.splitext(base_log_path)[1]

    # Get timestamp
    start_time = os.getenv("LOG_START_TIME", datetime.now().strftime("%Y%m%d_%H%M"))

    # Create timestamped path
    timestamped_path = f"{base_name}_{start_time}{ext}"

    return timestamped_path

save_file = False
def log_reward(completions, solution, **kwargs):
    """Combined reward function that calculates all rewards and handles logging."""
    global save_file
    if not os.getenv("DEBUG_MODE") == "true":
        return [0 for _ in range(len(solution))]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Use provided rewards
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
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if not save_file and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            save_file = True
            import wandb
            if wandb.run is not None:
                wandb.save(log_path, policy="live")
    return [0 for _ in range(len(solution))]

reward_funcs_registry = {
    "accuracy": iou_reward,
    "format": format_reward,
    "log": log_reward,
}

#%%

from trl.trainer import GRPOConfig, GRPOTrainer, ModelConfig
from vllm_grpo_trainer import VisionGRPOVLLMTrainer

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

model_config = ModelConfig(
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
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
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 1024,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 450,
    # save_steps = 450,
    # max_grad_norm = 0.5, # need to be the same as the gradient clipping in zero3.json
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
    deepspeed="zero3.json",
)


trainer = VisionGRPOVLLMTrainer(
    model = model,
    reward_funcs = list(reward_funcs_registry.values()),
    train_dataset = converted_dataset,
    args = training_args,
    peft_config = peft_config,
)
#%%
trainer_stats = trainer.train()
