
import os
import json
import torch
import numpy as np
from PIL import Image
import cv2
import logging
from torch.utils.data import Dataset
from transformers import AutoProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide

from .refer import REFER
from .utils import ANSWER_LIST


class UnifiedQWSADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor: AutoProcessor,
        image_size: int = 1024,
        max_seq_length: int = 512,
        split: str = "train",
        seg_token_idx: int = None,
        training_mode: str = "finetune",  # "finetune", "pretrain", "so_grpo"
        **kwargs
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.split = split
        self.seg_token_idx = seg_token_idx
        self.training_mode = training_mode
        self.kwargs = kwargs
        
        self.transform = ResizeLongestSide(image_size)
        
        # Mode-specific initialization
        if training_mode == "pretrain":
            self.mask_probability = kwargs.get("mask_probability", 0.3)
            self.data_format = kwargs.get("data_format", "image_text")
        elif training_mode == "so_grpo":
            self.num_prompts_per_step = kwargs.get("num_prompts_per_step", 8)
            self.prompt_templates = self._create_prompt_templates()
        elif training_mode == "finetune":
            self.data_format = kwargs.get("data_format", "reason_seg")
        
        # Load data samples
        self.data_samples = self._load_data_samples()
        logging.info(f"Loaded {len(self.data_samples)} samples for {training_mode} mode ({split})")
    
    def _load_data_samples(self):
        """Load data samples based on training mode"""
        if self.training_mode == "pretrain":
            return self._load_pretrain_data()
        elif self.training_mode == "so_grpo":
            return self._load_grpo_data()
        else:  # finetune
            return self._load_finetune_data()
    
    def _load_finetune_data(self):
        """Load finetune data"""
        data_format = self.kwargs.get("data_format", "reason_seg")
        data_file = os.path.join(self.data_path, f"{self.split}.json")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Finetune data file not found: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        return data if isinstance(data, list) else [data]
    
    def _load_pretrain_data(self):
        """Load pretrain data"""
        if os.path.isfile(self.data_path):
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                return data if isinstance(data, list) else [data]
            else:
                raise ValueError(f"Unsupported data file format: {self.data_path}")
        elif os.path.isdir(self.data_path):
            samples = []
            for file in os.listdir(self.data_path):
                if file.endswith('.json'):
                    file_path = os.path.join(self.data_path, file)
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                    samples.extend(file_data if isinstance(file_data, list) else [file_data])
            return samples
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
    
    def _load_grpo_data(self):
        """Load GRPO data"""
        data_file = os.path.join(self.data_path, f"{self.split}.json")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"GRPO data file not found: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        return data if isinstance(data, list) else [data]
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        """Get single training sample based on mode"""
        sample = self.data_samples[idx]
        
        try:
            if self.training_mode == "pretrain":
                return self._process_pretrain_sample(sample)
            elif self.training_mode == "so_grpo":
                return self._process_grpo_sample(sample)
            else:  # finetune
                return self._process_finetune_sample(sample)
        except Exception as e:
            logging.error(f"Error processing sample {idx}: {e}")
            return self._get_default_sample()
    
    def _process_finetune_sample(self, sample):
        """Process finetune sample"""
        data_format = self.kwargs.get("data_format", "reason_seg")
        
        if data_format == "reason_seg":
            return self._process_reason_seg_sample(sample)
        elif data_format == "ref_seg":
            return self._process_ref_seg_sample(sample)
        elif data_format == "vqa":
            return self._process_vqa_sample(sample)
        else:
            return self._process_custom_sample(sample)
    
    def _process_reason_seg_sample(self, sample):
        """Process reasoning segmentation sample"""
        image_path = sample.get("image", "")
        query = sample.get("query", "")
        outputs = sample.get("outputs", "")
        json_path = sample.get("json", "")
        
        # Build full path
        base_dir = os.path.dirname(self.data_path)
        full_image_path = os.path.join(base_dir, image_path)
        full_json_path = os.path.join(base_dir, json_path) if json_path else ""
        
        # Load image
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess image
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # Load masks
        masks = torch.zeros((1, image_rgb.shape[0], image_rgb.shape[1]))
        if full_json_path and os.path.exists(full_json_path):
            masks = self._load_masks_from_json(full_json_path, image_rgb.shape[:2])
        
        # Build conversation
        if self.split == "train":
            # Training includes complete answer
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                {"role": "assistant", "content": outputs}
            ]
        else:
            # Validation has empty assistant content
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                {"role": "assistant", "content": ""}
            ]
        
        # Process text
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        # Create labels
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Only calculate loss for assistant's reply
        if self.split == "train":
            assistant_start = len(self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}, {"role": "assistant", "content": ""}],
                tokenize=False,
                add_generation_prompt=False
            ))
            labels[0, :assistant_start] = -100
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "images": image_for_sam,
            "masks_list": masks,
            "label_list": torch.ones(masks.shape[1], masks.shape[2]) * 255,
            "resize_list": [resize],
        }
    
    def _process_ref_seg_sample(self, sample):
        """Process reference segmentation sample"""
        image_path = sample.get("image", "")
        phrase = sample.get("phrase", "")
        
        base_dir = os.path.dirname(self.data_path)
        full_image_path = os.path.join(base_dir, image_path)
        
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # Load masks
        masks = self._load_masks_from_ref_seg_sample(sample, image_rgb.shape[:2])
        
        # Build conversation
        query = f"Please segment: {phrase}"
        if self.split == "train":
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                {"role": "assistant", "content": f"I'll segment {phrase} for you. [SEG]"}
            ]
        else:
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                {"role": "assistant", "content": ""}
            ]
        
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        if self.split == "train":
            assistant_start = len(self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}, {"role": "assistant", "content": ""}],
                tokenize=False,
                add_generation_prompt=False
            ))
            labels[0, :assistant_start] = -100
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "images": image_for_sam,
            "masks_list": masks,
            "label_list": torch.ones(masks.shape[1], masks.shape[2]) * 255,
            "resize_list": [resize],
        }
    
    def _process_vqa_sample(self, sample):
        """Process VQA sample"""
        image_path = sample.get("image", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        
        base_dir = os.path.dirname(self.data_path)
        full_image_path = os.path.join(base_dir, image_path)
        
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # VQA tasks usually don't need masks
        masks = torch.zeros((1, image_rgb.shape[0], image_rgb.shape[1]))
        
        # Build conversation
        if self.split == "train":
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": answer}
            ]
        else:
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": ""}
            ]
        
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        if self.split == "train":
            assistant_start = len(self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}, {"role": "assistant", "content": ""}],
                tokenize=False,
                add_generation_prompt=False
            ))
            labels[0, :assistant_start] = -100
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "images": image_for_sam,
            "masks_list": masks,
            "label_list": torch.ones(masks.shape[1], masks.shape[2]) * 255,
            "resize_list": [resize],
        }
    
    def _process_custom_sample(self, sample):
        """Process custom sample"""
        # Process according to custom data format
        # Here's a basic framework, can be modified according to actual needs
        image_path = sample.get("image", "")
        text_input = sample.get("input", "")
        text_output = sample.get("output", "")
        
        base_dir = os.path.dirname(self.data_path)
        full_image_path = os.path.join(base_dir, image_path)
        
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # Check if masks are needed
        masks = torch.zeros((1, image_rgb.shape[0], image_rgb.shape[1]))
        if "mask" in sample and sample["mask"]:
            masks = self._load_custom_masks(sample["mask"], image_rgb.shape[:2])
        
        # Build conversation
        if self.split == "train":
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_input}]},
                {"role": "assistant", "content": text_output}
            ]
        else:
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_input}]},
                {"role": "assistant", "content": ""}
            ]
        
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        if self.split == "train":
            assistant_start = len(self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_input}]}, {"role": "assistant", "content": ""}],
                tokenize=False,
                add_generation_prompt=False
            ))
            labels[0, :assistant_start] = -100
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "images": image_for_sam,
            "masks_list": masks,
            "label_list": torch.ones(masks.shape[1], masks.shape[2]) * 255,
            "resize_list": [resize],
        }
    
    def _process_pretrain_sample(self, sample):
        """Process pretrain sample"""
        if self.data_format == "image_text":
            return self._process_image_text_sample(sample)
        elif self.data_format == "image_mask_text":
            return self._process_image_mask_text_sample(sample)
        else:  # image_only
            return self._process_image_only_sample(sample)
    
    def _process_image_text_sample(self, sample):
        """Process image-text pair sample"""
        image_path = sample.get("image_path", "")
        text = sample.get("text", "")
        
        # Load image
        if image_path.startswith('/'):
            full_image_path = image_path
        else:
            full_image_path = os.path.join(os.path.dirname(self.data_path), image_path)
        
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess image for SAM
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # Build conversation
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
            {"role": "assistant", "content": self._generate_response(text)}
        ]
        
        # Randomly decide whether to add segmentation task
        if np.random.random() < self.mask_probability:
            conversation[1]["content"] += " [SEG]"
            masks = self._generate_random_masks(image_rgb.shape[:2])
        else:
            masks = torch.zeros((1, image_rgb.shape[0], image_rgb.shape[1]))
        
        # Use processor to handle
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        # Create labels
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Only calculate loss for assistant's reply
        assistant_start = len(self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}, {"role": "assistant", "content": ""}],
            tokenize=False,
            add_generation_prompt=False
        ))
        labels[0, :assistant_start] = -100
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "images": image_for_sam,
            "masks_list": masks,
            "label_list": torch.ones(masks.shape[1], masks.shape[2]) * 255,  # ignore label
            "resize_list": [resize],
        }
    
    def _process_image_mask_text_sample(self, sample):
        """Process image-mask-text triplet sample"""
        image_path = sample.get("image_path", "")
        mask_path = sample.get("mask_path", "")
        text = sample.get("text", "")
        
        # Load image and mask
        full_image_path = os.path.join(os.path.dirname(self.data_path), image_path)
        full_mask_path = os.path.join(os.path.dirname(self.data_path), mask_path)
        
        image = cv2.imread(full_image_path)
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise FileNotFoundError(f"Cannot load image or mask: {full_image_path}, {full_mask_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # Process mask
        mask = (mask > 127).astype(np.float32)
        masks = torch.from_numpy(mask).unsqueeze(0)
        
        # Build conversation (including segmentation task)
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
            {"role": "assistant", "content": self._generate_response(text) + " [SEG]"}
        ]
        
        # Process text
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "images": image_for_sam,
            "masks_list": masks,
            "label_list": torch.ones(masks.shape[1], masks.shape[2]) * 255,
            "resize_list": [resize],
        }
    
    def _process_image_only_sample(self, sample):
        """Process pure image sample (for self-supervised learning)"""
        image_path = sample.get("image_path", "")
        
        full_image_path = os.path.join(os.path.dirname(self.data_path), image_path)
        image = cv2.imread(full_image_path)
        
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # Generate random text description
        text = "Describe this image in detail."
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
            {"role": "assistant", "content": self._generate_random_description()}
        ]
        
        inputs = self.processor(
            text=self.processor.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "images": image_for_sam,
            "masks_list": torch.zeros((1, image_rgb.shape[0], image_rgb.shape[1])),
            "label_list": torch.ones(image_rgb.shape[0], image_rgb.shape[1]) * 255,
            "resize_list": [resize],
        }
    
    def _process_grpo_sample(self, sample):
        """Process GRPO sample"""
        image_path = sample.get("image", "")
        objects = sample.get("objects", [])  # List of objects in image
        gt_masks = sample.get("masks", [])  # Ground truth masks
        
        # Build full path
        base_dir = os.path.dirname(self.data_path)
        full_image_path = os.path.join(base_dir, image_path)
        
        # Load image
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess image
        image_for_sam = self.transform.apply_image(image)
        resize = image_for_sam.shape[:2]
        image_for_sam = self._preprocess_image_for_sam(image_for_sam)
        
        # Process ground truth masks
        processed_gt_masks = []
        for mask_info in gt_masks:
            mask = self._parse_mask(mask_info, image_rgb.shape[:2])
            processed_gt_masks.append(mask)
        
        # Generate multiple prompts
        prompts = self._generate_prompts(objects)
        
        # Prepare tensors
        images_tensor = image_for_sam.unsqueeze(0).repeat(self.num_prompts_per_step, 1, 1, 1)
        gt_masks_tensor = torch.stack(processed_gt_masks).unsqueeze(0).repeat(self.num_prompts_per_step, 1, 1, 1)
        
        return {
            "images": images_tensor,
            "gt_masks": gt_masks_tensor,
            "prompts": prompts,
            "resize": [resize] * self.num_prompts_per_step,
            "image_path": full_image_path,
        }
    
    def _create_prompt_templates(self) -> list:
        """Create prompt templates"""
        templates = [
            # Segmentation prompts
            "Please segment {object} in this image.",
            "Identify and outline {object} shown in image.",
            "Create a mask for {object} in this image.",
            "Segment the {object} in this image.",
            
            # Reasoning prompts
            "What is the main {object} in this image? Please segment it.",
            "Analyze this image and identify {object}, then create a segmentation mask.",
            "Examine the image and provide a segmentation of {object}.",
            "Analyze this image and identify {object}, then create a segmentation mask.",
            
            # Complex prompts
            "In this image, I need to identify {object} for a specific task. Please provide a precise segmentation mask.",
            "For medical analysis purposes, please outline the {object} visible in this image.",
            "As part of a computer vision task, please create an accurate mask for the {object}.",
            "For medical analysis purposes, please outline the {object} visible in this image.",
        ]
        
        return templates
    
    def _generate_prompts(self, objects: list) -> list:
        """Generate multiple prompts for given objects"""
        if not objects:
            # Use generic prompts if no objects specified
            generic_prompts = [
                "Please segment the main object in this image.",
                "Identify and outline the most prominent object.",
                "Create a segmentation mask for the central object.",
                "Segment the main object in this image."
            ]
            return [
                {"text": prompt, "object": "main_object"}
                for prompt in generic_prompts[:self.num_prompts_per_step]
            ]
        
        # Generate prompts for each object
        prompts = []
        for i in range(self.num_prompts_per_step):
            # Randomly select object and template
            obj = objects[i % len(objects)]
            template = self.prompt_templates[i % len(self.prompt_templates)]
            
            # Fill template
            if isinstance(obj, str):
                obj_name = obj
            elif isinstance(obj, dict):
                obj_name = obj.get("name", "object")
            else:
                obj_name = "object"
            
            prompt = template.format(object=obj_name)
            prompts.append({
                "text": prompt,
                "object": obj_name,
                "object_info": obj
            })
        
        return prompts
    
    def _generate_response(self, text):
        """Generate response based on input text"""
        # Simple response generation logic, can be more complex in actual applications
        if "segment" in text.lower() or "mask" in text.lower():
            return "I can help you segment objects in this image. [SEG]"
        else:
            return "This image shows various visual elements that I can analyze and describe."
    
    def _generate_random_masks(self, image_shape):
        """Generate random masks for training"""
        h, w = image_shape
        num_masks = np.random.randint(1, 4)
        masks = np.zeros((num_masks, h, w), dtype=np.float32)
        
        for i in range(num_masks):
            # Generate random rectangular masks
            x1 = np.random.randint(0, w//4)
            y1 = np.random.randint(0, h//4)
            x2 = np.random.randint(3*w//4, w)
            y2 = np.random.randint(3*h//4, h)
            
            masks[i, y1:y2, x1:x2] = 1.0
        
        return torch.from_numpy(masks)
    
    def _generate_random_description(self):
        """Generate random image description"""
        descriptions = [
            "This image contains various objects and scenes that can be analyzed.",
            "The image shows visual elements with different colors and shapes.",
            "I can see multiple components in this image that form a coherent scene.",
            "This visual representation includes several interesting features worth noting."
        ]
        return np.random.choice(descriptions)
    
    def _preprocess_image_for_sam(self, image):
        """Preprocess image for SAM model"""
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image = (image - pixel_mean) / pixel_std
        
        _, h, w = image.shape
        padh = self.image_size - h
        padw = self.image_size - w
        image = torch.nn.functional.pad(image, (0, padw, 0, padh))
        
        return image
    
    def _load_masks_from_json(self, json_path, image_shape):
        """Load masks from JSON file"""
        try:
            with open(json_path, 'r') as f:
                mask_data = json.load(f)
            
            # Parse masks based on JSON format
            if isinstance(mask_data, list):
                # Multiple masks
                masks = []
                for mask_info in mask_data:
                    mask = self._parse_single_mask(mask_info, image_shape)
                    masks.append(mask)
                return torch.stack(masks, dim=0)
            else:
                # Single mask
                mask = self._parse_single_mask(mask_data, image_shape)
                return mask.unsqueeze(0)
        except Exception as e:
            logging.warning(f"Failed to load masks {json_path}: {e}")
            return torch.zeros((1, image_shape[0], image_shape[1]))
    
    def _parse_single_mask(self, mask_info, image_shape):
        """Parse single mask"""
        if isinstance(mask_info, dict):
            if "mask" in mask_info:
                # Mask is array format
                mask_array = np.array(mask_info["mask"])
                return torch.from_numpy(mask_array).float()
            elif "polygon" in mask_info:
                # Mask is polygon format
                return self._polygon_to_mask(mask_info["polygon"], image_shape)
            elif "bbox" in mask_info:
                # Mask is bounding box format
                return self._bbox_to_mask(mask_info["bbox"], image_shape)
        
        # Default return empty mask
        return torch.zeros(image_shape[0], image_shape[1])
    
    def _polygon_to_mask(self, polygon, image_shape):
        """Convert polygon to mask"""
        from PIL import Image, ImageDraw
        mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        
        if isinstance(polygon[0], list):
            # Multiple polygons
            for poly in polygon:
                draw.polygon(poly, fill=255)
        else:
            # Single polygon
            draw.polygon(polygon, fill=255)
        
        mask_array = np.array(mask)
        return torch.from_numpy(mask_array).float() / 255.0
    
    def _bbox_to_mask(self, bbox, image_shape):
        """Convert bounding box to mask"""
        h, w = image_shape
        mask = torch.zeros(h, w)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mask[y1:y2, x1:x2] = 1.0
        
        return mask
    
    def _load_masks_from_ref_seg_sample(self, sample, image_shape):
        """Load masks from reference segmentation sample"""
        if "mask" in sample:
            return self._load_masks_from_json(sample["mask"], image_shape)
        else:
            return torch.zeros((1, image_shape[0], image_shape[1]))
    
    def _load_custom_masks(self, mask_info, image_shape):
        """Load custom masks"""
        if isinstance(mask_info, str):
            # Mask file path
            return self._load_masks_from_json(mask_info, image_shape)
        else:
            # Mask data
            return self._load_masks_from_json(mask_info, image_shape)
    
    def _parse_mask(self, mask_info, image_shape):
        """Parse mask information"""
        if isinstance(mask_info, str):
            # Mask file path
            mask_path = os.path.join(os.path.dirname(self.data_path), mask_info)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
                return torch.from_numpy(mask)
            else:
                logging.warning(f"Mask file not found: {mask_path}")
                return torch.zeros(image_shape[0], image_shape[1])
        
        elif isinstance(mask_info, dict):
            # Mask data
            if "polygon" in mask_info:
                return self._polygon_to_mask(mask_info["polygon"], image_shape)
            elif "bbox" in mask_info:
                return self._bbox_to_mask(mask_info["bbox"], image_shape)
            elif "mask" in mask_info:
                mask_array = np.array(mask_info["mask"])
                return torch.from_numpy(mask_array).float()
            else:
                return torch.zeros(image_shape[0], image_shape[1])
        
        else:
            return torch.zeros(image_shape[0], image_shape[1])
    
    def _get_default_sample(self):
        """Get default sample for error handling"""
        dummy_image = torch.zeros(3, self.image_size, self.image_size)
        dummy_text = "This is a default sample."
        
        if self.training_mode == "so_grpo":
            dummy_prompts = [
                {"text": "Please segment the main object.", "object": "main_object"}
                for _ in range(self.num_prompts_per_step)
            ]
            dummy_gt_masks = torch.zeros(self.num_prompts_per_step, self.image_size, self.image_size)
            
            return {
                "images": dummy_image.unsqueeze(0).repeat(self.num_prompts_per_step, 1, 1, 1),
                "gt_masks": dummy_gt_masks,
                "prompts": dummy_prompts,
                "resize": [(self.image_size, self.image_size)] * self.num_prompts_per_step,
                "image_path": "",
            }
        else:
            inputs = self.processor(
                text=dummy_text,
                images=Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)),
                return_tensors="pt",
                padding=True
            )
            
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "input_ids": inputs["input_ids"].squeeze(0),
                "labels": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "images": dummy_image,
                "masks_list": torch.zeros((1, self.image_size, self.image_size)),
                "label_list": torch.ones(self.image_size, self.image_size) * 255,
                "resize_list": [(self.image_size, self.image_size)],
            }


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, processor=None
):
    # 1. Filter out invalid samples (those returning None in __getitem__)
    batch = [item for item in batch if item is not None]
    
    # 2. If entire batch is empty after filtering, return a complete but empty structure
    if not batch:
        return {
            "images": torch.tensor([], dtype=torch.float32),
            "pixel_values": torch.tensor([], dtype=torch.float32),
            "input_ids": torch.tensor([], dtype=torch.long),
            "labels": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.long),
            "masks_list": [],
            "label_list": [],
            "resize_list": [],
            "offset": torch.LongTensor([]),
            "inference": [],
            "image_paths": [],
            "questions_list": [],
            "sampled_classes_list": [],
            "image_grid_thw": None,
        }
    
    assert processor is not None, "Qwen requires processor to be passed to collate_fn"
    
    # Initialize collectors
    image_path_list, images_list, pil_images, all_messages_structured = [], [], [], []
    masks_list, label_list, resize_list, questions_list, sampled_classes_list = [], [], [], [], [], []
    offset_list = [0]
    cnt = 0
    inferences = []
    
    # 3. Collect raw data from valid batch
    for (
        image_path, images, pil_image, messages, masks, label, resize,
        questions, sampled_classes, inference
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        pil_images.append(pil_image)
        all_messages_structured.append(messages)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += 1
        offset_list.append(cnt)
        inferences.append(inference)
    
    # 4. Use processor to handle text and images
    # For training, conversation includes answer. For validation, assistant's part is empty.
    texts_for_processing = [
        processor.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=(not conversation[-1]['content'])
        )
        for conversation in all_messages_structured
    ]
    
    # 5. Package the batch using processor
    inputs = processor(
        text=texts_for_processing,
        images=pil_images,
        return_tensors="pt",
        padding=True
    )
    
    # 6. Extract all required outputs
    input_ids = inputs['input_ids']
    attention_masks = inputs['attention_mask']
    images_clip = inputs['pixel_values']
    image_grid_thw = inputs.get('image_grid_thw')
    
    # 7. Create target tensor for 'labels' and mask padding
    targets = input_ids.clone()
    targets[targets == tokenizer.pad_token_id] = -100
    
    # 8. Build and return final batch dictionary
    return_dict = {
        "images": torch.stack(images_list, dim=0),
        "pixel_values": images_clip,
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "inference": inferences[0] if inferences else False,  # Handle empty list case
        # Auxiliary info, filtered before model call
        "image_paths": image_path_list,
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
    }
    
    if image_grid_thw is not None:
        return_dict['image_grid_thw'] = image_grid_thw
    
    return return_dict


# Backward compatibility aliases
FinetuneDataset = UnifiedQWSADataset  # For existing code compatibility
PretrainDataset = UnifiedQWSADataset  # For existing code compatibility
GRPODataset = UnifiedQWSADataset      # For existing code compatibility
ValDataset = None  # Will be deprecated in favor of UnifiedQWSADataset
