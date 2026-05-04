#!/bin/bash

# =========================================================================
#                     QWSA Pretrain Launch Script
# =========================================================================

# --- 1. GPU Configuration ---
GPUS_TO_USE="0,1,2,3"
DEEPSPEED_PATH="/home/ubuntu/miniconda3/envs/lisa/bin/deepspeed"

# --- 2. Path Configuration ---
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
VISION_PRETRAINED="medsam_vit_b.pth"
DATA_PATH="/path/to/pretrain/data"  # Modify to actual data path
OUTPUT_DIR="./pretrain_output"
DEEPSPEED_CONFIG="configs/deepspeed_pretrain.json"

# --- 3. Training Parameters ---
EPOCHS=10
BATCH_SIZE=8
GRAD_ACCUM_STEPS=4
LEARNING_RATE=1e-4
MAX_SEQ_LENGTH=2048
IMAGE_SIZE=1024
PRECISION="bf16"

# --- 4. Stain Self-Distillation Parameters ---
USE_STAIN_DISTILLATION=true
STAIN_DISTILLATION_WEIGHT=0.1
STAIN_AUGMENTATION_PROB=0.5

# --- 5. Logging and Save Configuration ---
SAVE_INTERVAL=1000
LOG_INTERVAL=10

# =========================================================================
#                           Script Execution
# =========================================================================

echo "================================================="
echo "              QWSA Pretrain Launch Script"
echo "================================================="
echo "[Config] Model path: $MODEL_PATH"
echo "[Config] Vision pretrained weights: $VISION_PRETRAINED"
echo "[Config] Data path: $DATA_PATH"
echo "[Config] Output directory: $OUTPUT_DIR"
echo "[Config] Using GPUs: $GPUS_TO_USE"
echo "[Config] Using stain distillation: $USE_STAIN_DISTILLATION"
echo "-------------------------------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Construct DeepSpeed parameters
DEEPSPEED_INCLUDE_STR="localhost:$GPUS_TO_USE"

# Start pretraining using unified training script
echo "[Status] Starting pretraining..."

nohup $DEEPSPEED_PATH --include "$DEEPSPEED_INCLUDE_STR" --master_port=29500 \
    ../train.py \
    --mode="pretrain" \
    --model_name_or_path="$MODEL_PATH" \
    --vision_pretrained="$VISION_PRETRAINED" \
    --data_path="$DATA_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --epochs="$EPOCHS" \
    --batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM_STEPS" \
    --learning_rate="$LEARNING_RATE" \
    --max_seq_length="$MAX_SEQ_LENGTH" \
    --image_size="$IMAGE_SIZE" \
    --precision="$PRECISION" \
    --use_stain_distillation="$USE_STAIN_DISTILLATION" \
    --stain_distillation_weight="$STAIN_DISTILLATION_WEIGHT" \
    --stain_augmentation_prob="$STAIN_AUGMENTATION_PROB" \
    --save_interval="$SAVE_INTERVAL" \
    --log_interval="$LOG_INTERVAL" \
    --deepspeed_config="$DEEPSPEED_CONFIG" \
    > "$OUTPUT_DIR/pretrain_$(date +%m%d%H%M).log" 2>&1 &

echo "[Status] Pretraining started in background"
echo "[Tip] Use the following command to view training logs:"
echo "         tail -f $OUTPUT_DIR/pretrain_$(date +%m%d%H%M).log"
echo "================================================="