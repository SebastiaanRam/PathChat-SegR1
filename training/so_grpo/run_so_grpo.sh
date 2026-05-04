#!/bin/bash

# =========================================================================
#                     QWSA SO-GRPO Launch Script
# =========================================================================

# --- 1. GPU Configuration ---
GPUS_TO_USE="0,1,2,3"
DEEPSPEED_PATH="/home/ubuntu/miniconda3/envs/lisa/bin/deepspeed"

# --- 2. Path Configuration ---
MODEL_PATH="/path/to/finetuned/model"  # Modify to actual finetuned model path
VISION_PRETRAINED="medsam_vit_b.pth"
DATA_PATH="/path/to/grpo/data"  # Modify to actual GRPO data path
REWARD_MODEL_PATH="/path/to/reward/model"  # Modify to actual reward model path
OUTPUT_DIR="./grpo_output"
DEEPSPEED_CONFIG="configs/deepspeed_grpo.json"

# --- 3. Training Parameters ---
EPOCHS=10
STEPS_PER_EPOCH=1000
BATCH_SIZE=4
LEARNING_RATE=1e-6
MAX_SEQ_LENGTH=512
IMAGE_SIZE=1024
PRECISION="bf16"

# --- 4. GRPO Parameters ---
KL_COEF=0.1
CLIPRANGE=0.2
VF_COEF=0.5
ENT_COEF=0.01
GAMMA=0.99
LAM=0.95
MAX_GRAD_NORM=1.0

# --- 5. Reward Parameters ---
REWARD_SCALE=1.0
USE_DICE_REWARD=true
USE_IOU_REWARD=true
USE_TEXT_REWARD=true

# --- 6. Data Parameters ---
NUM_PROMPTS_PER_STEP=8

# --- 7. Logging and Save Configuration ---
SAVE_INTERVAL=500
LOG_INTERVAL=10
EVAL_INTERVAL=1000

# =========================================================================
#                           Script Execution
# =========================================================================

echo "================================================="
echo "              QWSA SO-GRPO Launch Script"
echo "================================================="
echo "[Config] Base model path: $MODEL_PATH"
echo "[Config] Vision pretrained weights: $VISION_PRETRAINED"
echo "[Config] GRPO data path: $DATA_PATH"
echo "[Config] Reward model path: $REWARD_MODEL_PATH"
echo "[Config] Output directory: $OUTPUT_DIR"
echo "[Config] Using GPUs: $GPUS_TO_USE"
echo "[Config] Learning rate: $LEARNING_RATE"
echo "[Config] KL coefficient: $KL_COEF"
echo "-------------------------------------------------"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Construct DeepSpeed parameters
DEEPSPEED_INCLUDE_STR="localhost:$GPUS_TO_USE"

# Start SO-GRPO training using unified training script
echo "[Status] Starting SO-GRPO training..."

nohup $DEEPSPEED_PATH --include "$DEEPSPEED_INCLUDE_STR" --master_port=29502 \
    ../train.py \
    --mode="so_grpo" \
    --model_name_or_path="$MODEL_PATH" \
    --vision_pretrained="$VISION_PRETRAINED" \
    --data_path="$DATA_PATH" \
    --reward_model_path="$REWARD_MODEL_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --epochs="$EPOCHS" \
    --steps_per_epoch="$STEPS_PER_EPOCH" \
    --batch_size="$BATCH_SIZE" \
    --learning_rate="$LEARNING_RATE" \
    --max_seq_length="$MAX_SEQ_LENGTH" \
    --image_size="$IMAGE_SIZE" \
    --precision="$PRECISION" \
    --kl_coef="$KL_COEF" \
    --cliprange="$CLIPRANGE" \
    --vf_coef="$VF_COEF" \
    --ent_coef="$ENT_COEF" \
    --gamma="$GAMMA" \
    --lam="$LAM" \
    --max_grad_norm="$MAX_GRAD_NORM" \
    --reward_scale="$REWARD_SCALE" \
    --use_dice_reward="$USE_DICE_REWARD" \
    --use_iou_reward="$USE_IOU_REWARD" \
    --use_text_reward="$USE_TEXT_REWARD" \
    --num_prompts_per_step="$NUM_PROMPTS_PER_STEP" \
    --save_interval="$SAVE_INTERVAL" \
    --log_interval="$LOG_INTERVAL" \
    --eval_interval="$EVAL_INTERVAL" \
    --deepspeed_config="$DEEPSPEED_CONFIG" \
    > "$OUTPUT_DIR/grpo_$(date +%m%d%H%M).log" 2>&1 &

echo "[Status] SO-GRPO training started in background"
echo "[Tip] Use following command to view training logs:"
echo "         tail -f $OUTPUT_DIR/grpo_$(date +%m%d%H%M).log"
echo "================================================="