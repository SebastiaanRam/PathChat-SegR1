#!/usr/bin/env python3

import os
import sys
import argparse
import logging

# Add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import shared utilities
from training.shared.config_manager import (
    add_base_arguments, add_finetune_arguments, 
    add_pretrain_arguments, add_grpo_arguments
)
from training.shared.training_utils import setup_logging


def create_unified_parser():
    """Create unified argument parser for all training modes"""
    parser = argparse.ArgumentParser(description="Unified QWSA Training Script")
    
    # Add mode selection
    parser.add_argument("--mode", required=True, choices=["finetune", "pretrain", "so_grpo"],
                       help="Training mode")
    
    # Add base arguments
    parser = add_base_arguments(parser)
    
    # Add mode-specific arguments in separate groups
    finetune_group = parser.add_argument_group('Finetune Arguments')
    finetune_group = add_finetune_arguments(finetune_group)
    
    pretrain_group = parser.add_argument_group('Pretrain Arguments')
    pretrain_group = add_pretrain_arguments(pretrain_group)
    
    grpo_group = parser.add_argument_group('GRPO Arguments')
    grpo_group = add_grpo_arguments(grpo_group)
    
    return parser


def main():
    # Parse arguments
    parser = create_unified_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.output_dir, args.local_rank)
    logging.info(f"Starting unified QWSA training in {args.mode} mode")
    
    # Route to appropriate training script
    if args.mode == "finetune":
        from training.finetune.finetune_qwsa import main as finetune_main
        finetune_main()
    elif args.mode == "pretrain":
        from training.pretrain.pretrain_qwsa import main as pretrain_main
        pretrain_main()
    elif args.mode == "so_grpo":
        from training.so_grpo.so_grpo_qwsa import main as grpo_main
        grpo_main()
    else:
        raise ValueError(f"Unknown training mode: {args.mode}")


if __name__ == "__main__":
    main()