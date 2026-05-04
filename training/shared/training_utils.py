"""
Shared training utilities for all training modes
"""

import os
import logging
import torch
import deepspeed
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoProcessor


def setup_logging(output_dir: str, local_rank: int = 0):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO if local_rank == 0 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def get_tokenizer_and_processor(model_name_or_path: str, trust_remote_code: bool = True):
    """Get tokenizer and processor for the model"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="right"
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, processor


def get_seg_token_idx(tokenizer, fallback_token: str = "<|seg_mask|>"):
    """Get the token index for [SEG] or fallback token"""
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    if seg_token_idx == tokenizer.unk_token_id:
        seg_token_idx = tokenizer.convert_tokens_to_ids(fallback_token)
        logging.warning(f"[SEG] token not found, using {fallback_token}, ID: {seg_token_idx}")
    return seg_token_idx


def create_deepspeed_config(args) -> Dict[str, Any]:
    """Create DeepSpeed configuration based on training mode"""
    base_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": getattr(args, 'gradient_accumulation_steps', 1),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": getattr(args, 'weight_decay', 0.0),
                "betas": (0.9, 0.95)
            }
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": getattr(args, 'max_grad_norm', 1.0),
    }
    
    # Add scheduler configuration
    if hasattr(args, 'warmup_steps'):
        base_config["scheduler"] = {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * getattr(args, 'total_steps_per_epoch', 1000),
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps
            }
        }
    
    # Add zero optimization based on training mode
    if hasattr(args, 'training_mode'):
        if args.training_mode == "pretrain":
            base_config["zero_optimization"] = {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
                "contiguous_gradients": True,
                "overlap_comm": True
            }
        elif args.training_mode == "finetune":
            base_config["zero_optimization"] = {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
                "contiguous_gradients": True,
                "overlap_comm": True
            }
        elif args.training_mode == "so_grpo":
            base_config["zero_optimization"] = {
                "stage": 1,
            }
            # Use fixed learning rate for GRPO
            base_config["scheduler"] = {
                "type": "LambdaLR",
                "params": {
                    "lr_lambda": lambda step: 1.0
                }
            }
    
    return base_config


def save_checkpoint(model_engine, output_dir: str, step: int, is_best: bool = False):
    """Save model checkpoint"""
    if is_best:
        checkpoint_dir = os.path.join(output_dir, "best_model")
    else:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    
    model_engine.save_checkpoint(checkpoint_dir)
    logging.info(f"Saved checkpoint to: {checkpoint_dir}")
    return checkpoint_dir


def validate_model(model, val_dataloader, tokenizer, args, global_step: int):
    """Common validation function"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            # Move data to GPU
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.cuda()
            
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_samples += batch['input_ids'].size(0)
    
    avg_loss = total_loss / len(val_dataloader)
    logging.info(f"Validation - Step {global_step}, Average Loss: {avg_loss:.4f}")
    
    model.train()
    return avg_loss


def apply_stain_augmentation(batch, stain_augmentor):
    """Apply stain augmentation to batch of images"""
    if stain_augmentor is None or "images" not in batch:
        return batch
    
    try:
        augmented_images = []
        for img in batch["images"]:
            # Convert to numpy array for augmentation
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            
            # Apply RandStainNA augmentation
            augmented_img_np = stain_augmentor.augment(img_np)
            
            # Convert back to tensor
            augmented_img = torch.from_numpy(augmented_img_np.transpose(2, 0, 1)).float() / 255.0
            if img.is_cuda:
                augmented_img = augmented_img.cuda()
            augmented_images.append(augmented_img)
        
        # Replace images in batch
        batch["images"] = torch.stack(augmented_images)
    except Exception as e:
        logging.warning(f"Stain augmentation failed, using original images: {e}")
        # If augmentation fails, continue with original images
        pass
    
    return batch