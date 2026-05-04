
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


class VisionEncoder(nn.Module):
    """Base vision encoder class"""
    
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.load_model(model_name, pretrained)
    
    def load_model(self, model_name, pretrained):
        """Load vision encoder model"""
        raise NotImplementedError
    
    def forward(self, images):
        """Forward pass through vision encoder"""
        raise NotImplementedError


class SamVisionEncoder(VisionEncoder):
    """SAM vision encoder for segmentation tasks"""
    
    def __init__(self, model_name="facebook/sam-vit-base", pretrained=True):
        super().__init__(model_name, pretrained)
        self.image_size = 1024
        self.patch_size = 16
        self.embed_dim = 768
    
    def load_model(self, model_name, pretrained):
        """Load SAM vision encoder"""
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if pretrained else torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def forward(self, images):
        """Extract features using SAM vision encoder"""
        if self.processor is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get image embeddings
            with torch.no_grad():
                outputs = self.model.vision_encoder(**inputs)
                image_embeddings = outputs.last_hidden_state
        
        return image_embeddings


class RuiPathVisionEncoder(VisionEncoder):
    """RuiPath vision encoder for pathology-specific features"""
    
    def __init__(self, model_name="paige-ai/RuiPath", pretrained=True):
        super().__init__(model_name, pretrained)
        self.image_size = 224  
        self.patch_size = 16
        self.embed_dim = 768
    
    def load_model(self, model_name, pretrained):
        """Load RuiPath vision encoder"""
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if pretrained else torch.float32
            )
        except Exception as e:
            print(f"Warning: Could not load RuiPath model: {e}")
            print("Falling back to a general pathology vision encoder...")
            # Fallback to a general vision model trained on medical images
            self.model = AutoModel.from_pretrained(
                "microsoft/BiomedVLP-CXR-BERT-specialized",
                trust_remote_code=True,
                torch_dtype=torch.float16 if pretrained else torch.float32
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def forward(self, images):
        """Extract pathology-specific features using RuiPath"""
        if self.processor is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get image embeddings with pathology-specific understanding
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeddings = outputs.last_hidden_state
        
        return image_embeddings


class StainInvariantEncoder(VisionEncoder):
    """Stain-invariant encoder with self-distillation"""
    
    def __init__(self, base_encoder, alpha=0.1):
        super().__init__(f"stain_invariant_{base_encoder}")
        self.base_encoder = base_encoder
        self.alpha = alpha  # Weight for stain variation penalty
        self.mae_head = nn.Linear(base_encoder.embed_dim, base_encoder.embed_dim * 64 * 64)  # For 16x16 patches
        
    def load_model(self, model_name, pretrained):
        """Initialize stain-invariant encoder"""
        pass
    
    def apply_stain_augmentation(self, images):
        """Apply RandStainNA augmentation in LAB color space"""
        # Convert RGB to LAB using proper color space conversion
        lab_images = self.rgb_to_lab(images)
        
        # Apply random channel-wise linear transformations
        batch_size = lab_images.shape[0]
        for i in range(batch_size):
            # Random transformation matrix
            transform = torch.randn(3, 3) * 0.1 + torch.eye(3)
            lab_images[i] = torch.matmul(lab_images[i], transform)
        
        # Convert back to RGB
        return self.lab_to_rgb(lab_images)
    
    def rgb_to_lab(self, rgb_images):
        """Convert RGB to LAB color space using OpenCV"""
        import cv2
        import numpy as np
        
        # Convert torch tensor to numpy
        if isinstance(rgb_images, torch.Tensor):
            rgb_np = rgb_images.cpu().numpy()
        else:
            rgb_np = rgb_images
            
        # Ensure RGB format and correct data type
        if rgb_np.max() <= 1.0:  # If normalized to [0,1]
            rgb_np = (rgb_np * 255).astype(np.uint8)
        
        lab_images = []
        for i in range(rgb_np.shape[0]):
            # Convert RGB to LAB using OpenCV
            if rgb_np[i].shape[-1] == 3:  # RGB image
                lab = cv2.cvtColor(rgb_np[i], cv2.COLOR_RGB2LAB)
            else:  # Grayscale or other format
                lab = cv2.cvtColor(rgb_np[i], cv2.COLOR_GRAY2LAB)
            lab_images.append(lab)
        
        lab_images = np.array(lab_images)
        
        # Convert back to torch tensor
        return torch.from_numpy(lab_images).to(rgb_images.device if isinstance(rgb_images, torch.Tensor) else 'cpu')
    
    def lab_to_rgb(self, lab_images):
        """Convert LAB to RGB color space using OpenCV"""
        import cv2
        import numpy as np
        
        # Convert torch tensor to numpy
        if isinstance(lab_images, torch.Tensor):
            lab_np = lab_images.cpu().numpy()
        else:
            lab_np = lab_images
            
        rgb_images = []
        for i in range(lab_np.shape[0]):
            # Convert LAB to RGB using OpenCV
            rgb = cv2.cvtColor(lab_np[i], cv2.COLOR_LAB2RGB)
            rgb_images.append(rgb)
        
        rgb_images = np.array(rgb_images)
        
        # Normalize back to [0,1] if input was normalized
        rgb_images = rgb_images.astype(np.float32) / 255.0
        
        # Convert back to torch tensor
        return torch.from_numpy(rgb_images).to(lab_images.device if isinstance(lab_images, torch.Tensor) else 'cpu')
    
    def compute_stain_loss(self, embeddings1, embeddings2, stain_vectors1, stain_vectors2):
        """Compute stain-invariant loss according to paper formula:"""
        # Cosine similarity between stain template vectors
        cos_sim = torch.nn.functional.cosine_similarity(stain_vectors1, stain_vectors2, dim=-1)
        
        # Feature difference between augmented views
        feat_diff = embeddings1 - embeddings2
        feat_diff_norm = torch.norm(feat_diff, dim=-1)
        
        # Weighted loss: cosine distance weighted by stain variation magnitude
        stain_loss = self.alpha * torch.mean(feat_diff_norm * (1 - cos_sim))
        return stain_loss
    
    def forward(self, images):
        """Forward pass with stain-invariant learning"""
        # Generate two stain-augmented views
        images_aug1 = self.apply_stain_augmentation(images)
        images_aug2 = self.apply_stain_augmentation(images)
        
        # Extract features from both views
        feat1 = self.base_encoder(images_aug1)
        feat2 = self.base_encoder(images_aug2)
        
        stain_vectors1 = self.extract_stain_vectors(images_aug1)
        stain_vectors2 = self.extract_stain_vectors(images_aug2)
        
        # Compute stain-invariant loss
        if self.training:
            stain_loss = self.compute_stain_loss(feat1, feat2, stain_vectors1, stain_vectors2)
            return feat1, feat2, stain_loss
        
        # During inference, use original images
        return self.base_encoder(images)
    
    def extract_stain_vectors(self, images):
        """Extract stain template vectors [A_v, D_v] from LAB space"""

        batch_size = images.shape[0]
        # Placeholder: return random stain vectors
        return torch.randn(batch_size, 2) 


def create_vision_encoder(encoder_type, model_name=None, pretrained=True, **kwargs):
    """Factory function to create vision encoder"""
    if encoder_type.lower() == "sam":
        return SamVisionEncoder(model_name or "facebook/sam-vit-base", pretrained)
    elif encoder_type.lower() == "ruipath":
        return RuiPathVisionEncoder(model_name or "paige-ai/RuiPath", pretrained)
    elif encoder_type.lower() == "stain_invariant":
        base_encoder = create_vision_encoder(kwargs.get("base_type", "sam"), model_name, pretrained)
        return StainInvariantEncoder(base_encoder, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")