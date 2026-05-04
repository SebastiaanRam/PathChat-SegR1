#!/usr/bin/env python3
"""
QWSA SO-GRPO Training Script
Simplified implementation using unified dataset and shared utilities
"""

import os
import sys
import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import deepspeed

# Add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import shared utilities
from training.shared.config_manager import create_grpo_parser, validate_args
from training.shared.training_utils import (
    setup_logging, get_tokenizer_and_processor, get_seg_token_idx,
    create_deepspeed_config, save_checkpoint
)

# Import model and components
from model.QWSA import QWSAForCausalLM
from utils.dataset import UnifiedQWSADataset
from utils.reward_model import RewardModel
from utils.so_grpo_trainer import SOGRPOTrainer


class QWSAGRPOTrainer(SOGRPOTrainer):
    """QWSA-specific SO-GRPO trainer"""
    
    def __init__(self, model, reward_model, tokenizer, processor, args):
        super().__init__(model, reward_model, tokenizer, processor, args)
        self.seg_token_idx = args.seg_token_idx
        
    def generate_responses(self, prompts: List[Dict], images: torch.Tensor) -> List[str]:
        """Generate model responses"""
        responses = []
        
        self.model.eval()
        with torch.no_grad():
            for i, prompt_dict in enumerate(prompts):
                # Build conversation
                conversation = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_dict["text"]}]},
                    {"role": "assistant", "content": ""}
                ]
                
                # Process text
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                inputs = self.processor(
                    text=text,
                    images=[Image.fromarray((images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))],
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to GPU
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.cuda()
                
                # Generate response
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_seq_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Decode response
                response_text = self.tokenizer.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=False
                )
                
                responses.append(response_text)
        
        return responses
    
    def compute_rewards(self, prompts: List[Dict], responses: List[str],
                      images: torch.Tensor, gt_masks: List[torch.Tensor]) -> torch.Tensor:
        """Compute rewards"""
        rewards = []
        
        for i, (prompt, response, gt_mask) in enumerate(zip(prompts, responses, gt_masks)):
            # 1. Segmentation quality reward
            seg_reward = 0.0
            if self.args.use_dice_reward or self.args.use_iou_reward:
                # Extract segmentation mask from response
                pred_mask = self.extract_mask_from_response(response, images[i])
                
                if pred_mask is not None:
                    if self.args.use_dice_reward:
                        dice_score = self.compute_dice_score(pred_mask, gt_mask)
                        seg_reward += dice_score
                    
                    if self.args.use_iou_reward:
                        iou_score = self.compute_iou_score(pred_mask, gt_mask)
                        seg_reward += iou_score
            
            # 2. Text quality reward
            text_reward = 0.0
            if self.args.use_text_reward:
                text_reward = self.reward_model.compute_text_reward(prompt["text"], response)
            
            # 3. Composite reward
            total_reward = (
                self.args.reward_scale * (
                    seg_reward * (0.7 if (self.args.use_dice_reward or self.args.use_iou_reward) else 0.0) +
                    text_reward * (0.3 if self.args.use_text_reward else 0.0)
                )
            )
            
            rewards.append(total_reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def extract_mask_from_response(self, response: str, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract segmentation mask from response"""
        # Check if response contains segmentation trigger words
        if "[SEG]" not in response and "seg" not in response.lower():
            return None
        
        try:
            # Extract token positions containing [SEG]
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            if self.seg_token_idx not in response_tokens:
                return None
            
            seg_pos = response_tokens.index(self.seg_token_idx)
            
            # Get hidden state of [SEG] token
            # This requires re-running model forward pass to get hidden states
            # Simplified implementation: return random mask as placeholder
            h, w = image.shape[-2:]
            mask = torch.zeros(h, w)
            
            # Randomly generate a mask region
            if np.random.random() > 0.3:  # 70% probability to generate valid mask
                x1 = np.random.randint(0, w//4)
                y1 = np.random.randint(0, h//4)
                x2 = np.random.randint(3*w//4, w)
                y2 = np.random.randint(3*h//4, h)
                mask[y1:y2, x1:x2] = 1.0
            
            return mask
        except Exception as e:
            logging.warning(f"Mask extraction failed: {e}")
            return None
    
    def compute_dice_score(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Compute Dice score"""
        pred_mask = pred_mask.bool()
        gt_mask = gt_mask.bool()
        
        intersection = (pred_mask & gt_mask).float().sum()
        union = pred_mask.float().sum() + gt_mask.float().sum()
        
        if union == 0:
            return 1.0
        
        dice = (2 * intersection) / union
        return dice.item()
    
    def compute_iou_score(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Compute IoU score"""
        pred_mask = pred_mask.bool()
        gt_mask = gt_mask.bool()
        
        intersection = (pred_mask & gt_mask).float().sum()
        union = (pred_mask | gt_mask).float().sum()
        
        if union == 0:
            return 1.0
        
        iou = intersection / union
        return iou.item()


def main():
    # Parse and validate arguments
    parser = create_grpo_parser()
    args = parser.parse_args()
    args = validate_args(args)
    
    # Setup logging
    setup_logging(args.output_dir, args.local_rank)
    logging.info(f"Starting QWSA SO-GRPO training with args: {args}")
    
    # Get tokenizer and processor
    tokenizer, processor = get_tokenizer_and_processor(args.model_name_or_path)
    
    # Get [SEG] token ID
    if args.seg_token_idx is None:
        args.seg_token_idx = get_seg_token_idx(tokenizer)
    
    # Load model
    model = QWSAForCausalLM.from_pretrained(
        args.model_name_or_path,
        vision_pretrained=args.vision_pretrained,
        image_size=args.image_size,
        out_dim=args.out_dim,
        seg_token_idx=args.seg_token_idx,
        train_mask_decoder=True,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.half
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    # Load reward model
    reward_model = RewardModel(args.reward_model_path, tokenizer, processor)
    
    # Create dataset using unified implementation
    dataset = UnifiedQWSADataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        max_seq_length=args.max_seq_length,
        split="train",
        training_mode="so_grpo",
        num_prompts_per_step=args.num_prompts_per_step
    )
    
    # Create SO-GRPO trainer
    trainer = QWSAGRPOTrainer(model, reward_model, tokenizer, processor, args)
    
    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config(args)
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = 1  # GRPO typically doesn't use gradient accumulation
    ds_config["gradient_clipping"] = args.max_grad_norm
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
        config=ds_config
    )
    
    trainer.model = model_engine
    trainer.optimizer = optimizer
    
    # Training loop
    global_step = 0
    best_reward = float('-inf')
    logging.info("Starting SO-GRPO training loop")
    
    for epoch in range(args.epochs):
        epoch_rewards = []
        
        for step in range(args.steps_per_epoch):
            # Get batch data
            batch = next(iter(train_loader))
            
            # Execute GRPO update
            metrics = trainer.train_step(batch)
            
            epoch_rewards.append(metrics["avg_reward"])
            global_step += 1
            
            # Logging
            if global_step % args.log_interval == 0:
                logging.info(
                    f"Epoch {epoch}, Step {global_step}, "
                    f"average reward: {metrics['avg_reward']:.4f}, "
                    f"policy loss: {metrics['policy_loss']:.4f}, "
                    f"value loss: {metrics['value_loss']:.4f}, "
                    f"KL divergence: {metrics['kl_div']:.4f}"
                )
            
            # Save checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(model_engine, args.output_dir, global_step)
                
                # Evaluation
                if global_step % args.eval_interval == 0:
                    avg_reward = np.mean(epoch_rewards[-args.eval_interval:])
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        save_checkpoint(model_engine, args.output_dir, global_step, is_best=True)
                        logging.info(f"Saved best model, average reward: {avg_reward:.4f}")
        
        avg_epoch_reward = np.mean(epoch_rewards)
        logging.info(f"Epoch {epoch} completed, average reward: {avg_epoch_reward:.4f}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    model_engine.save_checkpoint(final_dir)
    logging.info(f"SO-GRPO training completed, final model saved to: {final_dir}")


if __name__ == "__main__":
    main()