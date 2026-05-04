"""
Configuration manager for all training modes
"""

import argparse
import os
from typing import Dict, Any


def add_base_arguments(parser: argparse.ArgumentParser):
    """Add common arguments for all training modes"""
    # Model parameters
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Pretrained model path")
    parser.add_argument("--vision_pretrained", default="", type=str, help="SAM pretrained weight path")
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--out_dim", default=256, type=int)
    
    # Training parameters
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    
    # Technical parameters
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--log_interval", default=10, type=int)
    
    # DeepSpeed parameters
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--deepspeed_config", default="", type=str, help="DeepSpeed config file path")
    
    return parser


def add_finetune_arguments(parser: argparse.ArgumentParser):
    """Add finetune-specific arguments"""
    # Data parameters
    parser.add_argument("--data_path", required=True, type=str, help="Finetuning data path")
    parser.add_argument("--data_format", default="reason_seg", type=str,
                       choices=["reason_seg", "ref_seg", "vqa", "custom"])
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--seg_token_idx", type=int, help="[SEG] token ID")
    
    # Vision encoder parameters
    parser.add_argument("--use_ruipath_encoder", action="store_true", default=False,
                       help="Whether to use RuiPath pathology-specific vision encoder")
    parser.add_argument("--ruipath_model_path", default="", type=str,
                       help="RuiPath pretrained model path")
    
    # Stain invariance parameters
    parser.add_argument("--use_stain_invariant", action="store_true", default=False,
                       help="Whether to use stain-invariant encoder")
    parser.add_argument("--use_stain_augmentation", action="store_true", default=False,
                       help="Whether to use stain augmentation during training")
    parser.add_argument("--stain_augmentation_prob", default=0.3, type=float,
                       help="Stain augmentation probability")
    
    # Training parameters
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--eval_interval", default=1000, type=int)
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    
    # Loss weights
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    
    # Set training mode
    parser.set_defaults(training_mode="finetune")
    
    return parser


def add_pretrain_arguments(parser: argparse.ArgumentParser):
    """Add pretrain-specific arguments"""
    # Data parameters
    parser.add_argument("--data_path", required=True, type=str, help="Pretraining data path")
    parser.add_argument("--max_seq_length", default=2048, type=int)
    
    # Training parameters
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    
    # Stain self-distillation parameters
    parser.add_argument("--use_stain_distillation", action="store_true", default=True,
                       help="Whether to use stain self-distillation")
    parser.add_argument("--stain_distillation_weight", default=0.1, type=float,
                       help="Stain self-distillation loss weight")
    parser.add_argument("--stain_augmentation_prob", default=0.5, type=float,
                       help="Stain augmentation probability")
    
    # Set training mode
    parser.set_defaults(training_mode="pretrain")
    
    return parser


def add_grpo_arguments(parser: argparse.ArgumentParser):
    """Add GRPO-specific arguments"""
    # Data parameters
    parser.add_argument("--data_path", required=True, type=str, help="GRPO data path")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--seg_token_idx", type=int, help="[SEG] token ID")
    parser.add_argument("--num_prompts_per_step", default=8, type=int)
    
    # Training parameters
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--eval_interval", default=1000, type=int)
    
    # GRPO parameters
    parser.add_argument("--kl_coef", default=0.1, type=float)
    parser.add_argument("--cliprange", default=0.2, type=float)
    parser.add_argument("--vf_coef", default=0.5, type=float)
    parser.add_argument("--ent_coef", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--lam", default=0.95, type=float)
    
    # Reward model parameters
    parser.add_argument("--reward_model_path", required=True, type=str, help="Reward model path")
    parser.add_argument("--reward_scale", default=1.0, type=float)
    parser.add_argument("--use_dice_reward", action="store_true", default=True)
    parser.add_argument("--use_iou_reward", action="store_true", default=True)
    parser.add_argument("--use_text_reward", action="store_true", default=True)
    
    # Set training mode
    parser.set_defaults(training_mode="so_grpo")
    
    return parser


def create_finetune_parser():
    """Create argument parser for finetune mode"""
    parser = argparse.ArgumentParser(description="QWSA Finetuning")
    parser = add_base_arguments(parser)
    parser = add_finetune_arguments(parser)
    return parser


def create_pretrain_parser():
    """Create argument parser for pretrain mode"""
    parser = argparse.ArgumentParser(description="QWSA Pretraining")
    parser = add_base_arguments(parser)
    parser = add_pretrain_arguments(parser)
    return parser


def create_grpo_parser():
    """Create argument parser for GRPO mode"""
    parser = argparse.ArgumentParser(description="QWSA SO-GRPO Training")
    parser = add_base_arguments(parser)
    parser = add_grpo_arguments(parser)
    return parser


def validate_args(args):
    """Validate arguments"""
    if args.training_mode == "finetune":
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
        if args.use_ruipath_encoder and not args.ruipath_model_path:
            raise ValueError("ruipath_model_path is required when use_ruipath_encoder is True")
    
    elif args.training_mode == "pretrain":
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    elif args.training_mode == "so_grpo":
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
        if not os.path.exists(args.reward_model_path):
            raise FileNotFoundError(f"Reward model path not found: {args.reward_model_path}")
    
    return args


def get_default_config(training_mode: str) -> Dict[str, Any]:
    """Get default configuration for training mode"""
    base_config = {
        "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "vision_pretrained": "",
        "image_size": 1024,
        "out_dim": 256,
        "output_dir": f"./{training_mode}_output",
        "epochs": 5,
        "batch_size": 4,
        "learning_rate": 3e-5,
        "weight_decay": 0.0,
        "precision": "bf16",
        "workers": 4,
        "save_interval": 500,
        "log_interval": 10,
        "local_rank": 0,
        "deepspeed_config": "",
    }
    
    if training_mode == "finetune":
        finetune_config = {
            "data_format": "reason_seg",
            "max_seq_length": 512,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 500,
            "max_grad_norm": 1.0,
            "eval_interval": 1000,
            "use_ruipath_encoder": False,
            "use_stain_invariant": False,
            "use_stain_augmentation": False,
            "stain_augmentation_prob": 0.3,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,v_proj",
            "ce_loss_weight": 1.0,
            "dice_loss_weight": 0.5,
            "bce_loss_weight": 2.0,
        }
        base_config.update(finetune_config)
    
    elif training_mode == "pretrain":
        pretrain_config = {
            "max_seq_length": 2048,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "use_stain_distillation": True,
            "stain_distillation_weight": 0.1,
            "stain_augmentation_prob": 0.5,
        }
        base_config.update(pretrain_config)
    
    elif training_mode == "so_grpo":
        grpo_config = {
            "max_seq_length": 512,
            "steps_per_epoch": 1000,
            "max_grad_norm": 1.0,
            "eval_interval": 1000,
            "num_prompts_per_step": 8,
            "kl_coef": 0.1,
            "cliprange": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "reward_scale": 1.0,
            "use_dice_reward": True,
            "use_iou_reward": True,
            "use_text_reward": True,
        }
        base_config.update(grpo_config)
    
    return base_config