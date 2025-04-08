# CUDA_VISIBLE_DEVICES=2 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct | tee vllm.log &

if [ ${dp:-0} -eq 0 ]; then
    cmd="accelerate"
else
    cmd="debugpy --listen 5678 --wait-for-client $(which accelerate)"
fi

cuda_count=$(nvidia-smi -L | wc -l)

# $cmd launch --num_processes 2 --config_file deepspeed_zero3.yaml run_r1_grpo.py --config grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml

datetime=$(date +%Y%m%d_%H%M%S)
export LOG_PATH="logs/qwen2_5_grpo_${datetime}.jsonl"
$cmd launch --num_processes $((cuda_count - 1)) qwen2_5_grpo.py
