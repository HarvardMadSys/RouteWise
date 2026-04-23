#!/bin/bash

# --- Configuration ---
# Set this variable to "true" to use 8 GPUs (0-7) and tensor_parallel_size 8
# Set to "false" (or any other value) to use 4 GPUs (4,5,6,7) and tensor_parallel_size 4
use_all_gpu="${USE_ALL_GPU:-false}"

# Port - can be overridden by environment variable
PORT="${PORT:-8001}"

# Model path - can be overridden by environment variable
MODEL_PATH="${MODEL_PATH:-/home/user/models/Qwen/Qwen3-4B}"

# --- Determine GPU settings based on the variable ---
if [ "$use_all_gpu" = "true" ]; then
    visible_devices="0,1,2,3,4,5,6,7"
    tensor_parallel_size=8
    echo "Configured to use 8 GPUs: CUDA_VISIBLE_DEVICES=$visible_devices, tensor_parallel_size=$tensor_parallel_size"
else
    visible_devices="4,5,6,7"
    tensor_parallel_size=4
    echo "Configured to use 4 GPUs: CUDA_VISIBLE_DEVICES=$visible_devices, tensor_parallel_size=$tensor_parallel_size"
fi

# --- VLLM OpenAI API Server Command ---
echo "Starting VLLM OpenAI API server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"

CUDA_VISIBLE_DEVICES="$visible_devices" \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size "$tensor_parallel_size" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name hybrid-inference \
    --max-model-len 32768
