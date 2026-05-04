# PathChat-SegR1: Pathological Reasoning Segmentation Model

## Project Overview

PathChat-SegR1 is a reasoning segmentation model specifically designed for pathology images, enabling zero-shot generalization segmentation through text queries. The model combines pathology-specific visual encoders with segmentation-optimized reinforcement learning methods to handle out-of-domain tissue morphologies and new pathology types beyond training distributions.

## Key Features

- **Pathology-Specific Visual Encoder**: Uses RuiPath encoder specifically trained on pathology image features
- **Stain-Invariant Self-Distillation**: Improves robustness to staining variations through LAB color space augmentation and masked autoencoding
- **Segmentation-Optimized Reinforcement Learning (SO-GRPO)**: Extends standard GRPO with temporal credit assignment, differentiable reward approximation, and sparsity control to optimize segmentation timing
- **Large-Scale Pathology Dataset**: Contains 118,667 triplets of pathology image, ground-truth mask, query, and reasoning chain

## Project Structure

```
pathchat-seg/
├── app.py                 # Gradio web application entry point
├── chat.py               # Command-line interactive chat interface
├── test.py               # Model inference testing script
├── train_qs.py           # Main model training script
├── run.sh                # Training launch script
├── test.sh               # Testing script
├── rebutt_iclr.pdf       # Paper file
├── model/                # Model architecture related code
│   ├── QWSA.py          # QWSA model main architecture
│   ├── add_model.py      # Additional model components
│   ├── qwen/            # Qwen vision-language model related code
│   └── segment_anything/ # SAM (Segment Anything Model) related code
├── dataset/              # Dataset storage and processing
│   ├── DigestPath/       # DigestPath dataset
│   ├── Glas/            # GlaS dataset
│   └── patch_level6/    # Patch-level data
├── training/             # Training related scripts and configurations
│   ├── train.py         # General training script
│   ├── finetune/        # Fine-tuning related scripts
│   ├── pretrain/        # Pre-training related scripts
│   ├── so_grpo/         # SO-GRPO reinforcement learning training
│   └── shared/          # Shared training utilities
├── utils/                # Utility functions and auxiliary code
│   ├── dataset.py        # Dataset processing classes
│   ├── utils.py          # General utility functions
│   ├── vision_encoders.py # Vision encoders
│   └── ...              # Other utility files
├── psydata/              # Data preprocessing and annotation tools
│   ├── preprocess.py     # Data preprocessing
│   ├── final.py         # Final data processing
│   └── ...              # Other data processing scripts
└── runs/                 # Training logs and checkpoint storage
```

## Installation Instructions

### Environment Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA 11.0+
- Other dependencies in `psydata/environment.yml`

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/Misterursw/pathchat-seg.git
cd pathchat-seg
```

2. Create and activate Conda environment
```bash
conda env create -f psydata/environment.yml
conda activate pathchat-seg
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
## Hardware Requirements

**Note:** Our training utilized 8 H800 GPUs, each with 80GB of memory. Please be aware that the memory requirement is high.

## Usage

### 1. Batch Testing

```bash
python test.py --model-path <model_path> --image-path <image_path> --prompt "<your_prompt>" --sam-checkpoint <sam_checkpoint> --out-dir <output_directory>
```

### 3. Model Training

#### Pre-training
```bash
bash training/pretrain/run_pretrain.sh
```

#### Fine-tuning
```bash
bash training/finetune/run_finetune.sh
```

#### SO-GRPO Reinforcement Learning Training
```bash
bash training/so_grpo/run_so_grpo.sh
```

### 4. Distributed Training

Use `run.sh` for multi-GPU training:
```bash
# Modify GPU configuration in run.sh
GPUS_TO_USE="0,1,2,3"

# Launch training
bash run.sh
```

## Dataset Preparation

### Public Datasets

The project supports multiple public pathology datasets:
- Camelyon16/Camelyon17 (Breast cancer lymph node metastasis)
- CRAG (Colorectal cancer)
- DigestPath (Digestive system)
- GlaS (Colon glands)
- WSSS4LUAD (Lung adenocarcinoma)

### Dataset Examples

The dataset includes diverse pathology image examples, such as breast cancer lymph node metastases, colorectal adenocarcinoma tissue, and lung adenocarcinoma with various histologic patterns.

### Custom Datasets

1. Place images in the `dataset/` directory
2. Run data preprocessing scripts:
```bash
python psydata/preprocess.py
```
3. Generate training data:
```bash
python psydata/final.py
```

## Training Pipeline

1. **Pre-training**: VLM learns pathology-specific knowledge, MedSAM encoder undergoes stain-invariant self-distillation
2. **Supervised Fine-tuning**: Learns to align vision and language embeddings for reasoning segmentation
3. **SO-GRPO Reinforcement Learning**: Learns to determine when to generate `<SEG>` token for optimal segmentation results

## Performance Evaluation

Zero-shot evaluation on multiple pathology datasets shows 61% improvement over state-of-the-art segmentation models.

