
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import logging
import os
from typing import List, Dict, Tuple, Optional

class RewardModel(nn.Module):
    """
    Multi-dimensional reward model
    Evaluates:
    1. Segmentation quality (IoU, Dice)
    2. Text quality (fluency, relevance)
    3. Multi-modal consistency
    """
    
    def __init__(self, model_path: str, tokenizer, processor):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        
        
        if model_path and os.path.exists(model_path):
            self.reward_model = AutoModel.from_pretrained(model_path)
            logging.info(f"Load the pre-trained reward model: {model_path}")
        else:
            
            self.reward_model = None
            logging.warning("Pre-trained reward model not found, using rule-based rewards")
        
        
        self.lambda_soft = 0.3 
        self.lambda_sparse = 0.2  
        self.lambda_spatial = 0.1  
        self.lambda_format = 0.1 
        self.lambda_len = 0.01  
        
       
        self.text_encoder = AutoModel.from_pretrained(
            "distilbert-base-uncased", 
            output_hidden_states=True
        )
        
        
        self.seg_weight = 0.6
        self.text_weight = 0.3
        self.consistency_weight = 0.1
    
    def compute_text_reward(self, prompt: str, response: str) -> float:
        """Compute text quality reward"""
        if self.reward_model is not None:
            return self._neural_text_reward(prompt, response)
        else:
            return self._rule_based_text_reward(prompt, response)
    
    def _neural_text_reward(self, prompt: str, response: str) -> float:
        """Evaluate text quality using neural network"""
        # Encode text
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        response_tokens = self.tokenizer.encode(response, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            # Get text embeddings
            prompt_emb = self.text_encoder(**prompt_tokens).last_hidden_state.mean(dim=1)
            response_emb = self.text_encoder(**response_tokens).last_hidden_state.mean(dim=1)
            
            # Compute relevance score
            relevance = F.cosine_similarity(prompt_emb, response_emb, dim=1).item()
            
            # Compute fluency score (based on perplexity)
            # Simplified implementation: use response length and repetition
            fluency = self._compute_fluency_score(response)
            
            # Combined score
            text_score = 0.7 * relevance + 0.3 * fluency
            return max(0.0, text_score)  # Ensure non-negative
    
    def _rule_based_text_reward(self, prompt: str, response: str) -> float:
        """Rule-based text quality evaluation"""
        score = 0.0
        
        # 1. Reasonable length
        if 10 <= len(response) <= 200:
            score += 0.3
        elif len(response) > 200:
            score -= 0.1
        
        # 2. Contains segmentation-related keywords
        seg_keywords = ["segment", "mask", "outline", "boundary", "seg"]
        if any(keyword in response.lower() for keyword in seg_keywords):
            score += 0.4
        
        # 3. Answer relevance (simple keyword matching)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words)
        relevance = overlap / max(len(prompt_words), 1)
        score += 0.3 * relevance
        
        return max(0.0, score)
    
    def _compute_fluency_score(self, text: str) -> float:
        """Compute text fluency score"""
        # 1. Repetition penalty
        words = text.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_penalty = 1.0 - (len(words) - unique_words) / len(words)
        else:
            repetition_penalty = 0.0
        
        # 2. Reasonable length
        length_score = 1.0
        if len(text) < 5:
            length_score = 0.5
        elif len(text) > 500:
            length_score = 0.7
        
        # Combined fluency score
        fluency = 0.6 * repetition_penalty + 0.4 * length_score
        return fluency
    
    def compute_seg_reward(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        """Compute segmentation quality reward"""
        if pred_mask is None or gt_mask is None:
            return 0.0
        
        # Ensure masks are binary
        pred_mask = (pred_mask > 0.5).float()
        gt_mask = gt_mask.float()
        
        # Compute IoU
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection
        
        if union == 0:
            return 1.0
        
        iou = intersection / union
        
        # Compute Dice coefficient
        dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum())
        
        # Combined score
        seg_score = 0.6 * iou + 0.4 * dice
        return seg_score.item()
    
    def compute_consistency_reward(self, text: str, mask: torch.Tensor,
                              image_features: torch.Tensor) -> float:
        """Compute multi-modal consistency reward"""
        
        
        mask_area = mask.sum().item()
        image_area = mask.numel()
        mask_ratio = mask_area / image_area
        
        # Extract size-related keywords from text
        size_keywords = {
            "small": 0.1, "tiny": 0.05, "little": 0.1,
            "medium": 0.3, "moderate": 0.3,
            "large": 0.6, "big": 0.6, "huge": 0.8
        }
        
        expected_ratio = 0.3  # Default expected ratio
        for keyword, ratio in size_keywords.items():
            if keyword in text.lower():
                expected_ratio = ratio
                break
        
        # Compute consistency score
        consistency = 1.0 - abs(mask_ratio - expected_ratio)
        return max(0.0, consistency)
    
    def compute_composite_reward(self, prompt: str, response: str,
                              pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                              image_features: torch.Tensor) -> float:
        """Compute composite reward"""
        # 1. Segmentation quality reward
        seg_reward = self.compute_seg_reward(pred_mask, gt_mask)
        
        # 2. Text quality reward
        text_reward = self.compute_text_reward(prompt, response)
        
        # 3. Multi-modal consistency reward
        consistency_reward = self.compute_consistency_reward(
            response, pred_mask, image_features
        )
        
        # 4. Composite reward
        total_reward = (
            self.seg_weight * seg_reward +
            self.text_weight * text_reward +
            self.consistency_weight * consistency_reward
        )
        
        return total_reward
    
    def extract_mask_from_response(self, response: str, image: torch.Tensor = None) -> torch.Tensor:
        """
        Extract segmentation mask from response
        """
        # Check if response contains segmentation trigger words
        if "<SEG>" not in response and "[SEG]" not in response and "seg" not in response.lower():
            return None
        
        try:
            # If image is provided, use image dimensions
            if image is not None:
                if image.dim() == 3:  # [C, H, W]
                    h, w = image.shape[-2:]
                elif image.dim() == 4:  # [B, C, H, W]
                    h, w = image.shape[-2:]
                else:
                    h, w = 256, 256  # Default size
            else:
                h, w = 256, 256  # Default size
            
            # Create a mask based on response content
            mask = torch.zeros(h, w)
            
            if np.random.random() > 0.3: 
                # Generate mask based on position information in response
                if "left" in response.lower():
                    x1, x2 = 0, w // 2
                elif "right" in response.lower():
                    x1, x2 = w // 2, w
                elif "center" in response.lower() or "middle" in response.lower():
                    x1, x2 = w // 4, 3 * w // 4
                else:
                    x1, x2 = w // 4, 3 * w // 4
                
                if "top" in response.lower() or "upper" in response.lower():
                    y1, y2 = 0, h // 2
                elif "bottom" in response.lower() or "lower" in response.lower():
                    y1, y2 = h // 2, h
                else:
                    y1, y2 = h // 4, 3 * h // 4
                
                mask[y1:y2, x1:x2] = 1.0
            
            return mask
        except Exception as e:
            logging.warning(f"Mask extraction failed: {e}")
            return None
    
    def compute_soft_rewards(self, responses: List[str], images: torch.Tensor,
                           gt_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute differentiable segmentation quality rewards
        """
        batch_size = len(responses)
        device = images.device if images is not None else torch.device('cpu')
        soft_rewards = torch.zeros(batch_size, device=device)
        
        for i, (response, gt_mask) in enumerate(zip(responses, gt_masks)):
            # Extract predicted mask from response
            pred_mask = self.extract_mask_from_response(response, images[i] if images is not None else None)
            
            if pred_mask is not None and gt_mask is not None:
                # Ensure masks are on the same device
                pred_mask = pred_mask.to(device)
                gt_mask = gt_mask.to(device)
                
                pred_soft = torch.sigmoid(pred_mask)
                gt_soft = gt_mask.float()
                
                # Flatten tensors for computation
                pred_flat = pred_soft.view(-1)
                gt_flat = gt_soft.view(-1)
                
                # Compute numerator and denominator according to paper formula
                epsilon = 1e-7
                numerator = 2.0 * torch.sum(pred_flat * gt_flat) + epsilon
                denominator = torch.sum(pred_flat) + torch.sum(gt_flat) + epsilon
                
                # Soft Dice coefficient
                soft_dice = numerator / denominator
                
                # Compute IoU
                intersection = torch.sum(pred_flat * gt_flat)
                union = torch.sum(pred_flat) + torch.sum(gt_flat) - intersection
                soft_iou = (intersection + epsilon) / (union + epsilon)
                
                soft_rewards[i] = 0.7 * soft_dice + 0.3 * soft_iou
        
        return soft_rewards
    
    def compute_sparse_rewards(self, prompts: List[str], responses: List[str],
                             seg_positions: List[int]) -> torch.Tensor:
        """
        Compute sparsity-aware rewards
        """
        batch_size = len(prompts)
        sparse_rewards = torch.zeros(batch_size)
        
        # Paper parameters for sparsity-aware rewards
        beta_sparse = 1.0  # Positive reward weight for spatial reasoning
        gamma_sparse = 0.5  # Negative reward weight for non-spatial reasoning
        
        for i, (prompt, response, seg_pos) in enumerate(zip(prompts, responses, seg_positions)):
            # Check if response contains spatial information 
            spatial_keywords = ["boundary", "region", "area", "location", "position", "center",
                              "left", "right", "top", "bottom", "upper", "lower"]
            has_spatial_info = any(keyword in response.lower() for keyword in spatial_keywords)
            
            # Check if <SEG> generation timing is reasonable 
            reasonable_timing = seg_pos > 10 and seg_pos < len(response.split()) * 0.8
            
            # Determine if current state is in S_spatial 
            is_spatial_state = has_spatial_info and reasonable_timing
            
            if is_spatial_state:
                sparse_rewards[i] = beta_sparse  # Positive reward for spatial reasoning
            else:
                sparse_rewards[i] = -gamma_sparse  # Negative reward for non-spatial reasoning
        
        return sparse_rewards
    
    def compute_spatial_rewards(self, prompts: List[str], responses: List[str],
                              images: torch.Tensor, gt_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute spatial-based rewards
        """
        batch_size = len(prompts)
        spatial_rewards = torch.zeros(batch_size)
        
        for i, (prompt, response, gt_mask) in enumerate(zip(prompts, responses, gt_masks)):
            # Extract location keywords
            location_keywords = self.extract_location_keywords(prompt)
            if not location_keywords:
                continue
            
            # Extract predicted mask from response
            pred_mask = self.extract_mask_from_response(response, images[i] if images is not None else None)
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
        """
        Compute format correctness rewards
        """
        batch_size = len(responses)
        format_rewards = torch.zeros(batch_size)
        
        for i, response in enumerate(responses):
            # Check if contains <SEG> marker
            if "<SEG>" in response or "[SEG]" in response:
                format_rewards[i] = 1.0
            else:
                format_rewards[i] = -1.0
        
        return format_rewards
    
    def compute_length_penalties(self, responses: List[str], grace_threshold: int = 50) -> torch.Tensor:
        """
        Compute length penalties
        """
        batch_size = len(responses)
        length_penalties = torch.zeros(batch_size)
        
        for i, response in enumerate(responses):
            response_length = len(response.split())
            if response_length > grace_threshold:
                # Logarithmic penalty
                penalty = -self.lambda_len * np.log(response_length - grace_threshold + 1)
                length_penalties[i] = penalty
        
        return length_penalties
    
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

class MultiTaskRewardModel(RewardModel):
    
    def __init__(self, model_path: str, tokenizer, processor):
        super().__init__(model_path, tokenizer, processor)
        
    
        self.task_weights = {
            "reason_seg": {"seg": 0.7, "text": 0.2, "consistency": 0.1},
            "ref_seg": {"seg": 0.8, "text": 0.1, "consistency": 0.1},
            "vqa": {"seg": 0.0, "text": 0.8, "consistency": 0.2},
            "general": {"seg": 0.6, "text": 0.3, "consistency": 0.1}
        }
    
    def compute_task_specific_reward(self, task_type: str, prompt: str, response: str,
                                 pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                                 image_features: torch.Tensor) -> float:
    
        weights = self.task_weights.get(task_type, self.task_weights["general"])
        
        
        seg_reward = self.compute_seg_reward(pred_mask, gt_mask)
        text_reward = self.compute_text_reward(prompt, response)
        consistency_reward = self.compute_consistency_reward(response, pred_mask, image_features)
        
        
        total_reward = (
            weights["seg"] * seg_reward +
            weights["text"] * text_reward +
            weights["consistency"] * consistency_reward
        )
        
        return total_reward