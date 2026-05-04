#!/usr/bin/env python3


import os
import sys
import logging
import torch
import deepspeed
from peft import LoraConfig, get_peft_model

# Add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import shared utilities
from training.shared.config_manager import create_finetune_parser, validate_args
from training.shared.training_utils import (
    setup_logging, get_tokenizer_and_processor, get_seg_token_idx,
    create_deepspeed_config, save_checkpoint, validate_model, apply_stain_augmentation
)

# Import model and components
from model.QWSA import QWSAForCausalLM
from utils.dataset import UnifiedQWSADataset
from utils.vision_encoders import RuiPathVisionEncoder, StainInvariantEncoder
from utils.randstainna import RandStainNA


def setup_model(args, tokenizer):
    """Setup model and LoRA configuration"""
    # Load base model
    model = QWSAForCausalLM.from_pretrained(
        args.model_name_or_path,
        vision_pretrained=args.vision_pretrained,
        image_size=args.image_size,
        out_dim=args.out_dim,
        seg_token_idx=args.seg_token_idx,
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
        train_mask_decoder=True,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.half
    )
    
    # Setup vision encoder
    if args.use_ruipath_encoder:
        logging.info("Using RuiPath pathology-specific vision encoder")
        ruipath_encoder = RuiPathVisionEncoder(
            model_path=args.ruipath_model_path,
            image_size=args.image_size,
            out_dim=args.out_dim
        )
        
        # Replace model's vision encoder
        if hasattr(model, 'vision_encoder'):
            model.vision_encoder = ruipath_encoder
            logging.info("Replaced with RuiPath encoder")
    
    if args.use_stain_invariant:
        logging.info("Using stain-invariant encoder")
        stain_encoder = StainInvariantEncoder(
            model_name_or_path=args.model_name_or_path,
            image_size=args.image_size,
            out_dim=args.out_dim
        )
        
        # Set as additional attribute of the model
        model.stain_encoder = stain_encoder
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(','),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to specific modules
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def main():
    # Parse and validate arguments
    parser = create_finetune_parser()
    args = parser.parse_args()
    args = validate_args(args)
    
    # Setup logging
    setup_logging(args.output_dir, args.local_rank)
    logging.info(f"Starting QWSA finetuning with args: {args}")
    
    # Get tokenizer and processor
    tokenizer, processor = get_tokenizer_and_processor(args.model_name_or_path)
    
    # Get [SEG] token ID
    if args.seg_token_idx is None:
        args.seg_token_idx = get_seg_token_idx(tokenizer)
    
    # Setup model
    model = setup_model(args, tokenizer)
    
    # Initialize stain augmentor (if needed)
    stain_augmentor = None
    if args.use_stain_augmentation:
        stain_augmentor = RandStainNA(
            std_adjust=False,
            probability=args.stain_augmentation_prob,
            distribution="normal"
        )
        logging.info(f"Initialized stain augmentor with probability: {args.stain_augmentation_prob}")
    
    # Create datasets using unified implementation
    train_dataset = UnifiedQWSADataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        max_seq_length=args.max_seq_length,
        split="train",
        seg_token_idx=args.seg_token_idx,
        training_mode="finetune",
        data_format=args.data_format
    )
    
    val_dataset = UnifiedQWSADataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        max_seq_length=args.max_seq_length,
        split="val",
        seg_token_idx=args.seg_token_idx,
        training_mode="finetune",
        data_format=args.data_format
    )
    
    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config(args)
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["scheduler"]["params"]["total_num_steps"] = args.epochs * (
        len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps)
    )
    ds_config["scheduler"]["params"]["warmup_num_steps"] = args.warmup_steps
    ds_config["gradient_clipping"] = args.max_grad_norm
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_dataset,
        config=ds_config
    )
    
    # Create validation data loader
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda batch: unified_collate_fn(batch, tokenizer, processor=processor)
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    logging.info("Starting finetuning loop")
    
    for epoch in range(args.epochs):
        model_engine.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Apply stain augmentation (if enabled)
            if args.use_stain_augmentation:
                batch = apply_stain_augmentation(batch, stain_augmentor)
            
            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % args.log_interval == 0:
                logging.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")
            
            # Validation
            if global_step % args.eval_interval == 0:
                val_loss = validate_model(model_engine, val_loader, tokenizer, args, global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model_engine, args.output_dir, global_step, is_best=True)
                    logging.info(f"Saved best model, validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(model_engine, args.output_dir, global_step)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch} completed, average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    model_engine.save_checkpoint(final_dir)
    logging.info(f"Finetuning completed, final model saved to: {final_dir}")


def unified_collate_fn(batch, tokenizer, processor=None):
    """Unified collate function for finetune mode"""
    from utils.dataset import collate_fn
    return collate_fn(batch, tokenizer=tokenizer, processor=processor)


if __name__ == "__main__":
    main()