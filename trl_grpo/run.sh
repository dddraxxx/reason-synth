# CUDA_VISIBLE_DEVICES=2 trl vllm-serve --model Qwen/Qwen2.5-3B-Instruct | tee vllm.log &
export RUN_ID=$(date +"%Y%m%d_%H%M%S")

if [ ${dp:-0} -eq 0 ]; then
    cmd="accelerate"
else
    cmd="debugpy --listen 5678 --wait-for-client $(which accelerate)"
fi

cuda_count=$(nvidia-smi -L | wc -l)

# $cmd launch --num_processes 2 --config_file deepspeed_zero3.yaml run_r1_grpo.py --config grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
$cmd launch --num_processes $((cuda_count - 1)) "qwen2_5_vl_(7b)_vision.py" --config zero3.json
