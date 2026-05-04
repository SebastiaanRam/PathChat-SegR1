#!/bin/bash

# =========================================================================
#                     QWSA Finetune Launch Script
# =========================================================================

# --- 1. GPU Configuration ---
GPUS_TO_USE="0,1,2,3"
DEEPSPEED_PATH="/home/ubuntu/miniconda3/envs/lisa/bin/deepspeed"

# --- 2. Path Configuration ---
MODEL_PATH="/path/to/pretrained/model"  # Modify to actual pretrained model path
VISION_PRETRAINED="medsam_vit_b.pth"
DATA_PATH="/path/to/finetune/data"  # Modify to actual finetune data path
OUTPUT_DIR="./finetune_output"
DEEPSPEED_CONFIG="configs/deepspeed_finetune.json"

# --- 3. Training Parameters ---
EPOCHS=5
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
LEARNING_RATE=3e-5
MAX_SEQ_LENGTH=512
IMAGE_SIZE=1024
PRECISION="bf16"

# --- 4. LoRA Parameters ---
USE_LORA=true
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj"

# --- 5. Loss Weights ---
CE_LOSS_WEIGHT=1.0
DICE_LOSS_WEIGHT=0.5
BCE_LOSS_WEIGHT=2.0

# --- 6. Logging and Save Configuration ---
SAVE_INTERVAL=500
LOG_INTERVAL=10
EVAL_INTERVAL=1000

# --- 7. Data Format ---
DATA_FORMAT="reason_seg"  # Options: reason_seg, ref_seg, vqa, custom

# =========================================================================
#                           Script Execution
# =========================================================================

echo "================================================="
echo "              QWSA Finetune Launch Script"
echo "================================================="
echo "[Config] Pretrained model path: $MODEL_PATH"
echo "[Config] Vision pretrained weights: $VISION_PRETRAINED"
echo "[Config] Finetune data path: $DATA_PATH"
echo "[Config] Output directory: $OUTPUT_DIR"
echo "[Config] Data format: $DATA_FORMAT"
echo "[Config] Using GPUs: $GPUS_TO_USE"
echo "[Config] Using LoRA: $USE_LORA"
echo "-------------------------------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Construct DeepSpeed parameters
DEEPSPEED_INCLUDE_STR="localhost:$GPUS_TO_USE"

# Start finetuning using unified training script
echo "[Status] Starting finetuning..."

nohup $DEEPSPEED_PATH --include "$DEEPSPEED_INCLUDE_STR" --master_port=29501 \
    ../train.py \
    --mode="finetune" \
    --model_name_or_path="$MODEL_PATH" \
    --vision_pretrained="$VISION_PRETRAINED" \
    --data_path="$DATA_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --data_format="$DATA_FORMAT" \
    --epochs="$EPOCHS" \
    --batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM_STEPS" \
    --learning_rate="$LEARNING_RATE" \
    --max_seq_length="$MAX_SEQ_LENGTH" \
    --image_size="$IMAGE_SIZE" \
    --precision="$PRECISION" \
    --use_lora="$USE_LORA" \
    --lora_r="$LORA_R" \
    --lora_alpha="$LORA_ALPHA" \
    --lora_dropout="$LORA_DROPOUT" \
    --lora_target_modules="$LORA_TARGET_MODULES" \
    --ce_loss_weight="$CE_LOSS_WEIGHT" \
    --dice_loss_weight="$DICE_LOSS_WEIGHT" \
    --bce_loss_weight="$BCE_LOSS_WEIGHT" \
    --save_interval="$SAVE_INTERVAL" \
    --log_interval="$LOG_INTERVAL" \
    --eval_interval="$EVAL_INTERVAL" \
    --deepspeed_config="$DEEPSPEED_CONFIG" \
    > "$OUTPUT_DIR/finetune_$(date +%m%d%H%M).log" 2>&1 &

echo "[Status] Finetuning started in background"
echo "[Tip] Use the following command to view training logs:"
echo "         tail -f $OUTPUT_DIR/finetune_$(date +%m%d%H%M).log"
echo "================================================="