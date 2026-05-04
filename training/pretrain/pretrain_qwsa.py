#!/usr/bin/env python3

import os
import sys
import logging
import torch
import torch.nn.functional as F
import deepspeed

# Add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import shared utilities
from training.shared.config_manager import create_pretrain_parser, validate_args
from training.shared.training_utils import (
    setup_logging, get_tokenizer_and_processor,
    create_deepspeed_config, save_checkpoint, apply_stain_augmentation
)

# Import model and components
from model.QWSA import QWSAForCausalLM
from utils.dataset import UnifiedQWSADataset
from utils.vision_encoders import StainInvariantEncoder
from utils.randstainna import RandStainNA


def main():
    # Parse and validate arguments
    parser = create_pretrain_parser()
    args = parser.parse_args()
    args = validate_args(args)
    
    # Setup logging
    setup_logging(args.output_dir, args.local_rank)
    logging.info(f"Starting QWSA pretraining with args: {args}")
    
    # Get tokenizer and processor
    tokenizer, processor = get_tokenizer_and_processor(args.model_name_or_path)
    
    # Load model
    model = QWSAForCausalLM.from_pretrained(
        args.model_name_or_path,
        vision_pretrained=args.vision_pretrained,
        image_size=args.image_size,
        out_dim=args.out_dim,
        seg_token_idx=0,  # Can be set to default value during pretraining
        train_mask_decoder=True,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.half
    )
    
    # Initialize stain-invariant encoder (if stain self-distillation is enabled)
    stain_encoder = None
    stain_augmentor = None
    
    if args.use_stain_distillation:
        logging.info("Initializing stain-invariant encoder and stain augmentor")
        
        # Initialize stain-invariant encoder
        stain_encoder = StainInvariantEncoder(
            model_name_or_path=args.model_name_or_path,
            image_size=args.image_size,
            out_dim=args.out_dim
        )
        
        # Initialize stain augmentor
        stain_augmentor = RandStainNA(
            std_adjust=False,
            probability=args.stain_augmentation_prob,
            distribution="normal"
        )
        
        # Move stain encoder to same device
        if torch.cuda.is_available():
            stain_encoder = stain_encoder.cuda()
    
    # Create dataset using unified implementation
    train_dataset = UnifiedQWSADataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        max_seq_length=args.max_seq_length,
        split="train",
        training_mode="pretrain",
        mask_probability=0.3,  # Default mask probability for pretraining
        data_format="image_text"  # Default data format for pretraining
    )
    
    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config(args)
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = args.warmup_steps
    ds_config["gradient_clipping"] = args.max_grad_norm
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_dataset,
        config=ds_config
    )
    
    # Training loop
    global_step = 0
    logging.info("Starting pretraining loop")
    
    for epoch in range(args.epochs):
        model_engine.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            # Stain self-distillation loss (if enabled)
            if args.use_stain_distillation and stain_encoder is not None:
                stain_loss = 0.0
                
                # Get original images
                if "images" in batch:
                    original_images = batch["images"]
                    
                    # Apply stain augmentation
                    augmented_batch = apply_stain_augmentation(batch, stain_augmentor)
                    augmented_images = augmented_batch["images"]
                    
                    # Compute stain-invariant features
                    with torch.no_grad():
                        original_features = stain_encoder.encode_features(original_images)
                        augmented_features = stain_encoder.encode_features(augmented_images)
                    
                    # Compute stain self-distillation loss (feature consistency)
                    stain_loss = F.mse_loss(original_features, augmented_features)
                    
                    # Total loss = original loss + stain self-distillation loss
                    loss = loss + args.stain_distillation_weight * stain_loss
            
            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % args.log_interval == 0:
                if args.use_stain_distillation and 'stain_loss' in locals():
                    logging.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}, "
                                f"Stain Loss: {stain_loss.item():.4f}")
                else:
                    logging.info(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")
            
            # Save checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(model_engine, args.output_dir, global_step)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch} completed, average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    model_engine.save_checkpoint(final_dir)
    logging.info(f"Pretraining completed, final model saved to: {final_dir}")


if __name__ == "__main__":
    main()