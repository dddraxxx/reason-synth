# %%capture
import logging
import os
os.environ["WANDB_PROJECT"] = "reason-synth"
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import json


#%%
from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")

dataset
#%%
dataset[2]["image"]
#%%
dataset[2]["text"]
#%%
from IPython.display import display, Math, Latex

latex = dataset[2]["text"]
display(Math(latex))

#%%
instruction = "Write the LaTeX representation for this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "prompt" : conversation }
convert_to_conversation(dataset[2])

#%% Let's convert the dataset into the "correct" format for finetuning:
random_indices = random.sample(range(len(dataset)), 200)
converted_dataset = [convert_to_conversation(dataset[i]) for i in random_indices]

#%% We look at how the conversations are structured for the first example:
converted_dataset[0]
#%%
def format_reward_func(completions, prompts, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []
    return [0] * len(completions)

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
peft_config = get_peft_config(model_config)

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 1e-6,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 512,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 450,
    save_steps = 450,
    max_grad_norm = 0.5, # need to be the same as the gradient clipping in zero3.json
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
    deepspeed="zero3.json",
)


trainer = VisionGRPOVLLMTrainer(
    model = model,
    reward_funcs = [format_reward_func],
    train_dataset = converted_dataset,
    args = training_args,
    peft_config = peft_config,
)
#%%
trainer_stats = trainer.train()
