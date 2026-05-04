
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class SOGRPOTrainer:

    def __init__(self, model, reward_model, tokenizer, processor, args):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.args = args
        
        # Policy network parameters
        self.policy_model = model
        self.value_model = model  # Usually shared parameters
        
        # SO-GRPO specific parameters
        self.gamma = getattr(args, 'gamma', 0.99)  # Discount factor
        self.lambda_gae = getattr(args, 'lambda_gae', 0.95)  # GAE parameter
        self.lambda_soft = getattr(args, 'lambda_soft', 0.3)  # Soft reward weight
        self.lambda_sparse = getattr(args, 'lambda_sparse', 0.2)  # Sparsity weight
        self.lambda_spatial = getattr(args, 'lambda_spatial', 0.1)  # Spatial reward weight
        self.lambda_format = getattr(args, 'lambda_format', 0.1)  # Format reward weight
        self.lambda_len = getattr(args, 'lambda_len', 0.01)  # Length penalty weight
        self.lambda_kl = getattr(args, 'lambda_kl', 0.01)  # KL divergence weight
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: args.learning_rate / (1 + getattr(args, 'eta', 1e-4) * step)
        )
        
        # Training statistics
        self.global_step = 0
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.kl_divs = []
        
        # SO-GRPO specific statistics
        self.segment_timings = []  # Record <SEG> token generation timing
        self.spatial_alignments = []  # Spatial alignment scores
        self.gradient_variances = []  # Gradient variances
    
    def train_step(self, batch: Dict) -> Dict:
        """
        Execute one step of SO-GRPO training
        Args:
            batch: Batch data containing prompts, images, gt_masks, etc.
        
        Returns:
            Dictionary containing various losses and metrics
        """
        # 1. Generate multiple responses
        prompts = batch["prompts"]
        images = batch["images"]
        gt_masks = batch["gt_masks"]
        
        responses, seg_token_positions = self.generate_responses_with_timing(prompts, images)
        
        # 2. Compute multi-dimensional rewards
        reward_components = self.compute_comprehensive_rewards(
            prompts, responses, images, gt_masks, seg_token_positions
        )
        
        # 3. Use GAE to compute time-step level advantages
        advantages, returns = self.compute_gae_advantages(reward_components["total"])
        
        # 4. Compute old policy probabilities
        with torch.no_grad():
            old_log_probs, old_values = self.evaluate_prompts(prompts, responses, images)
        
        # 5. Re-compute current policy probabilities
        current_log_probs, current_values = self.evaluate_prompts(prompts, responses, images)
        
        # 6. Compute SO-GRPO losses
        policy_loss, value_loss, kl_div = self.compute_so_grpo_losses(
            current_log_probs, old_log_probs,
            current_values, old_values,
            advantages, returns,
            reward_components
        )
        
        # 7. Compute total loss (including all reward components)
        # According to paper: J_SO-GRPO(θ) = E[Σ_t Â_GAE(s_t,a_t) logπ_θ(a_t|s_t)] + λ_soft·R_soft + λ_sparse·R_sparse + λ_spatial·R_spatial + λ_format·R_format + R_len - λ_KL·L_KL
        total_loss = (
            policy_loss
            - self.args.vf_coef * value_loss
            - self.args.ent_coef * self.compute_entropy(current_log_probs)
            + self.lambda_kl * kl_div
            + self.lambda_soft * reward_components["soft"]  # Differentiable segmentation reward (positive in paper)
            + self.lambda_sparse * reward_components["sparse"]  # Sparsity reward (positive in paper)
            + self.lambda_spatial * reward_components["spatial"]  # Spatial reward (positive in paper)
            + self.lambda_format * reward_components["format"]  # Format reward (positive in paper)
            + self.lambda_len * reward_components["length"]  # Length penalty (negative in paper)
        )
        
        # 8. Backpropagation and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Record gradient variance (for convergence analysis)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.args.max_grad_norm
        )
        self.gradient_variances.append(grad_norm.item())
        
        self.optimizer.step()
        self.scheduler.step()
        
        # 9. Record statistics
        self.episode_rewards.extend(reward_components["total"].tolist())
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.kl_divs.append(kl_div.item())
        self.segment_timings.extend(seg_token_positions)
        self.spatial_alignments.append(reward_components["spatial"].mean().item())
        
        self.global_step += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_div": kl_div.item(),
            "avg_reward": reward_components["total"].mean().item(),
            "soft_reward": reward_components["soft"].mean().item(),
            "sparse_reward": reward_components["sparse"].mean().item(),
            "spatial_reward": reward_components["spatial"].mean().item(),
            "format_reward": reward_components["format"].mean().item(),
            "length_penalty": reward_components["length"].mean().item(),
            "total_loss": total_loss.item(),
            "avg_seg_timing": np.mean(seg_token_positions),
            "gradient_variance": grad_norm.item(),
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
    
    def generate_responses_with_timing(self, prompts: List[Dict], images: torch.Tensor) -> Tuple[List[str], List[int]]:
        """Generate responses and record <SEG> token positions"""
        responses = []
        seg_positions = []
        
        self.policy_model.eval()
        with torch.no_grad():
            for i, (prompt, image) in enumerate(zip(prompts, images)):
                # Build complete conversation
                conversation = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt["text"]}]},
                ]
                
                # Process text
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                inputs = self.processor(
                    text=text,
                    images=[self._tensor_to_pil_image(image)],
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to GPU
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.cuda()
                
                # Generate response
                generated_tokens = []
                seg_token_id = self.tokenizer.encode("<SEG>", add_special_tokens=False)[0]
                
                for step in range(self.args.max_length):
                    outputs = self.policy_model(**inputs)
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                    generated_tokens.append(next_token.item())
                    
                    if next_token.item() == seg_token_id:
                        seg_positions.append(step)
                        break
                    
                    # Update input
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.unsqueeze(0)], dim=1)
                
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                responses.append(response)
        
        return responses, seg_positions
    
    def compute_comprehensive_rewards(self, prompts: List[Dict], responses: List[str],
                                 images: torch.Tensor, gt_masks: List[torch.Tensor],
                                 seg_positions: List[int]) -> Dict[str, torch.Tensor]:
        """Compute SO-GRPO multi-dimensional rewards"""
        batch_size = len(prompts)
        device = images.device
        
        # 1. Differentiable segmentation quality reward
        soft_rewards = self.compute_soft_rewards(responses, images, gt_masks)
        
        # 2. Sparsity-aware reward
        sparse_rewards = self.compute_sparse_rewards(prompts, responses, seg_positions)
        
        # 3. Spatial-based reward
        spatial_rewards = self.compute_spatial_rewards(prompts, responses, images, gt_masks)
        
        # 4. Format correctness reward
        format_rewards = self.compute_format_rewards(responses)
        
        # 5. Length penalty
        length_penalties = self.compute_length_penalties(responses)
        
        # 6. Total reward
        total_rewards = (
            self.lambda_soft * soft_rewards +
            self.lambda_sparse * sparse_rewards +
            self.lambda_spatial * spatial_rewards +
            self.lambda_format * format_rewards +
            self.lambda_len * length_penalties
        )
        
        return {
            "soft": soft_rewards,
            "sparse": sparse_rewards,
            "spatial": spatial_rewards,
            "format": format_rewards,
            "length": length_penalties,
            "total": total_rewards
        }
    
    def compute_soft_rewards(self, responses: List[str], images: torch.Tensor,
                         gt_masks: List[torch.Tensor]) -> torch.Tensor:
        """Compute differentiable segmentation quality rewards"""
        # Use reward model to compute differentiable Dice/IoU
        pred_masks = []
        for response in responses:
            # Extract predicted mask from response
            pred_mask = self.reward_model.extract_mask_from_response(response)
            pred_masks.append(pred_mask)
        
        # Convert to tensor
        pred_masks = torch.stack(pred_masks).to(images.device)
        gt_masks_tensor = torch.stack(gt_masks).to(images.device)
        
        # Compute soft Dice loss (differentiable)
        pred_soft = torch.sigmoid(pred_masks)
        gt_soft = gt_masks_tensor.float()
        
        intersection = (pred_soft * gt_soft).sum(dim=(1, 2, 3))
        union = pred_soft.sum(dim=(1, 2, 3)) + gt_soft.sum(dim=(1, 2, 3))
        
        soft_dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        soft_iou = intersection / (union - intersection + 1e-7)
        
        # Combined reward
        soft_rewards = 0.7 * soft_dice + 0.3 * soft_iou
        return soft_rewards
    
    def compute_sparse_rewards(self, prompts: List[Dict], responses: List[str],
                          seg_positions: List[int]) -> torch.Tensor:
        """Compute sparsity-aware rewards"""
        batch_size = len(prompts)
        sparse_rewards = torch.zeros(batch_size)
        
        for i, (prompt, response, seg_pos) in enumerate(zip(prompts, responses, seg_positions)):
            # Check if response contains spatial information
            spatial_keywords = ["boundary", "region", "area", "location", "position", "center"]
            has_spatial_info = any(keyword in response.lower() for keyword in spatial_keywords)
            
            # Check if <SEG> generation timing is reasonable
            reasonable_timing = seg_pos > 10 and seg_pos < len(response.split()) * 0.8
            
            if has_spatial_info and reasonable_timing:
                sparse_rewards[i] = 1.0  # Positive reward
            elif not has_spatial_info:
                sparse_rewards[i] = -0.5  # Negative reward
            else:
                sparse_rewards[i] = -0.2  # Slight negative reward
        
        return sparse_rewards
    
    def compute_spatial_rewards(self, prompts: List[Dict], responses: List[str],
                           images: torch.Tensor, gt_masks: List[torch.Tensor]) -> torch.Tensor:
        """Compute spatial-based rewards"""
        batch_size = len(prompts)
        spatial_rewards = torch.zeros(batch_size)
        
        for i, (prompt, response, gt_mask) in enumerate(zip(prompts, responses, gt_masks)):
            # Extract location keywords
            location_keywords = self.extract_location_keywords(prompt["text"])
            if not location_keywords:
                continue
            
            # Extract predicted mask from response
            pred_mask = self.reward_model.extract_mask_from_response(response)
            if pred_mask is None:
                continue
            
            # Compute mask centroids
            pred_center = self.compute_mask_centroid(pred_mask)
            gt_center = self.compute_mask_centroid(gt_mask)
            
            # Check if centroids match location keywords
            if self.is_location_consistent(location_keywords, pred_center, gt_center):
                spatial_rewards[i] = 1.0
            else:
                spatial_rewards[i] = -0.3
        
        return spatial_rewards
    
    def compute_format_rewards(self, responses: List[str]) -> torch.Tensor:
        """Compute format correctness rewards"""
        batch_size = len(responses)
        format_rewards = torch.zeros(batch_size)
        
        for i, response in enumerate(responses):
            # Check if contains <SEG> marker
            if "<SEG>" in response:
                format_rewards[i] = 1.0
            else:
                format_rewards[i] = -1.0
        
        return format_rewards
    
    def compute_length_penalties(self, responses: List[str]) -> torch.Tensor:
        """Compute length penalties"""
        batch_size = len(responses)
        length_penalties = torch.zeros(batch_size)
        
        grace_threshold = getattr(self.args, 'grace_threshold', 50)
        
        for i, response in enumerate(responses):
            response_length = len(response.split())
            if response_length > grace_threshold:
                # Logarithmic penalty
                penalty = -self.lambda_len * np.log(response_length - grace_threshold + 1)
                length_penalties[i] = penalty
        
        return length_penalties
    
    def compute_gae_advantages(self, rewards: torch.Tensor, values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use GAE to compute time-step level advantage
        """
        rewards_np = rewards.cpu().numpy()
        
        # If no value estimates provided, use rewards as simplified value estimates
        if values is None:
            values_np = rewards_np
        else:
            values_np = values.cpu().numpy()
        
        batch_size, seq_len = rewards_np.shape
        
        # Compute GAE advantages and returns
        advantages = np.zeros_like(rewards_np)
        returns = np.zeros_like(rewards_np)
        
        # Compute GAE independently for each batch according to paper formula
        for b in range(batch_size):
            gae = 0
            # Compute GAE from back to front using: Â_GAE(s_t,a_t) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
            for t in reversed(range(seq_len)):
                # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
                if t < seq_len - 1:
                    delta = rewards_np[b, t] + self.gamma * values_np[b, t + 1] - values_np[b, t]
                else:
                    delta = rewards_np[b, t] - values_np[b, t]  # Terminal state
                
                # GAE computation: GAE_t = δ_t + γλ * GAE_{t+1}
                # This implements the infinite sum: Σ_{l=0}^∞ (γλ)^l δ_{t+l}
                gae = delta + self.gamma * self.lambda_gae * gae
                advantages[b, t] = gae
                
                # Compute returns: R_t = r_t + γ * R_{t+1}
                if t < seq_len - 1:
                    returns[b, t] = rewards_np[b, t] + self.gamma * returns[b, t + 1]
                else:
                    returns[b, t] = rewards_np[b, t]
        
        # Apply variance reduction formula for better convergence
        # This helps stabilize training by reducing variance in advantage estimates
        variance_reduction = np.sqrt((1 - (self.gamma * self.lambda_gae)**(2 * seq_len)) /
                                  (1 - (self.gamma * self.lambda_gae)**2 + 1e-8))
        advantages *= variance_reduction
        
        # Standardize advantages (within batch only) for better training stability
        advantages_flat = advantages.flatten()
        if advantages_flat.std() > 1e-8:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        advantages = advantages_flat.reshape(advantages.shape)
        
        return torch.from_numpy(advantages).to(rewards.device), torch.from_numpy(returns).to(rewards.device)
    
    def compute_so_grpo_losses(self, current_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                              current_values: torch.Tensor, old_values: torch.Tensor,
                              advantages: torch.Tensor, returns: torch.Tensor,
                              reward_components: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SO-GRPO specific losses"""
        # Standard PPO loss
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(current_values, returns)
        
        # KL divergence
        kl_div = (old_log_probs - current_log_probs).mean()
        
        return policy_loss, value_loss, kl_div
    
    def evaluate_prompts(self, prompts: List[Dict], responses: List[str],
                       images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate policy probabilities and values for given prompts and responses"""
        log_probs = []
        values = []
        
        self.policy_model.eval()
        with torch.no_grad():
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                # Build complete conversation
                conversation = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt["text"]}]},
                    {"role": "assistant", "content": response}
                ]
                
                # Process text
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                inputs = self.processor(
                    text=text,
                    images=[self._tensor_to_pil_image(images[i])],
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to GPU
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.cuda()
                
                # Forward pass
                outputs = self.policy_model(**inputs, output_hidden_states=True)
                
                # Compute token probabilities
                logits = outputs.logits[:, -1, :]  # Last token logits
                probs = F.softmax(logits, dim=-1)
                
                # Get probability of response tokens
                response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                if len(response_tokens) > 0:
                    response_probs = probs[0, response_tokens].prod()
                else:
                    response_probs = torch.tensor(1.0)
                
                log_probs.append(torch.log(response_probs + 1e-8))
                
                # Get value estimate (using mean of hidden states)
                if outputs.hidden_states is not None:
                    value = outputs.hidden_states[-1].mean(dim=1).mean()
                else:
                    value = torch.tensor(0.0)
                values.append(value)
        
        return torch.stack(log_probs), torch.stack(values)
    
    def compute_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute policy entropy"""
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()
    
    def extract_location_keywords(self, text: str) -> List[str]:
        """Extract location keywords from text"""
        location_words = ["left", "right", "top", "bottom", "center", "upper", "lower",
                       "corner", "edge", "boundary", "region"]
        return [word for word in location_words if word in text.lower()]
    
    def compute_mask_centroid(self, mask: torch.Tensor) -> Tuple[float, float]:
        """Compute mask centroid"""
        if mask.sum() == 0:
            return (0.0, 0.0)
        
        y_coords, x_coords = torch.where(mask > 0.5)
        if len(y_coords) == 0:
            return (0.0, 0.0)
        
        centroid_y = y_coords.float().mean().item()
        centroid_x = x_coords.float().mean().item()
        return (centroid_x, centroid_y)
    
    def is_location_consistent(self, keywords: List[str], pred_center: Tuple[float, float],
                           gt_center: Tuple[float, float]) -> bool:
        """Check location consistency"""
        if not keywords:
            return True
        
        # Simplified implementation: check centroid distance
        distance = np.sqrt((pred_center[0] - gt_center[0])**2 + (pred_center[1] - gt_center[1])**2)
        return distance < 50  # Adjustable threshold
    
    def _tensor_to_pil_image(self, image_tensor: torch.Tensor):
        """Convert tensor to PIL image"""
        from PIL import Image
        import numpy as np
        
        # De-normalize
        if image_tensor.dim() == 3:  # [C, H, W]
            image = image_tensor.cpu().numpy().transpose(1, 2, 0)
        else:  # [B, C, H, W]
            image = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # Ensure values are in [0,255] range
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image)
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "avg_policy_loss": np.mean(self.policy_losses) if self.policy_losses else 0,
            "avg_value_loss": np.mean(self.value_losses) if self.value_losses else 0,
            "avg_kl_div": np.mean(self.kl_divs) if self.kl_divs else 0,
            "avg_seg_timing": np.mean(self.segment_timings) if self.segment_timings else 0,
            "avg_spatial_alignment": np.mean(self.spatial_alignments) if self.spatial_alignments else 0,
            "gradient_variance": np.mean(self.gradient_variances) if self.gradient_variances else 0,
            "convergence_trend": self._compute_convergence_trend(self.episode_rewards),
            "total_steps": self.global_step
        }
    
    def _compute_convergence_trend(self, rewards: List[float]) -> float:
        """Compute convergence trend"""
        if len(rewards) < 20:
            return 0.0
        
        # Compute difference between recent and early average rewards
        recent_avg = np.mean(rewards[-10:])
        early_avg = np.mean(rewards[:10])
        
        trend = (recent_avg - early_avg) / (abs(early_avg) + 1e-8)
        return trend