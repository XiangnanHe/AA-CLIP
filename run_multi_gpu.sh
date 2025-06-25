#!/bin/bash

# Multi-GPU Training Script for AA-CLIP

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "CUDA not available. Please install CUDA drivers."
    exit 1
fi

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $NUM_GPUS GPU(s)"

# Default parameters
DATASET=${1:-"VisA"}
SHOT=${2:-32}
SAVE_PATH=${3:-"ckpt/multi_gpu_${DATASET}"}
TEXT_EPOCH=${4:-5}
IMAGE_EPOCH=${5:-20}

echo "Training on dataset: $DATASET"
echo "Shot: $SHOT"
echo "Save path: $SAVE_PATH"
echo "Text epochs: $TEXT_EPOCH"
echo "Image epochs: $IMAGE_EPOCH"

# Method 1: DataParallel (Simpler, single machine)
echo "=== Method 1: DataParallel Training ==="
python train.py \
    --dataset $DATASET \
    --shot $SHOT \
    --save_path "${SAVE_PATH}_dataparallel" \
    --text_epoch $TEXT_EPOCH \
    --image_epoch $IMAGE_EPOCH \
    --text_batch_size $((16 * NUM_GPUS)) \
    --image_batch_size $((2 * NUM_GPUS))

# Method 2: DistributedDataParallel (Better performance)
echo "=== Method 2: DistributedDataParallel Training ==="
python train_distributed.py \
    --dataset $DATASET \
    --shot $SHOT \
    --save_path "${SAVE_PATH}_distributed" \
    --text_epoch $TEXT_EPOCH \
    --image_epoch $IMAGE_EPOCH \
    --text_batch_size 16 \
    --image_batch_size 2 \
    --world_size $NUM_GPUS

echo "Training completed!"
echo "Results saved in:"
echo "  - DataParallel: ${SAVE_PATH}_dataparallel"
echo "  - Distributed: ${SAVE_PATH}_distributed" 