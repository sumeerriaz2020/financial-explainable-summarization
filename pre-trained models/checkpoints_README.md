# Pre-trained Model Checkpoints

Complete guide for downloading and using pre-trained model checkpoints.

---

## 📦 Available Checkpoints

| Checkpoint | Size | Description | Download |
|------------|------|-------------|----------|
| **Base Model** | 1.6 GB | BART-large + initialized components | [HuggingFace](#base-model-checkpoint) |
| **Fully Trained** | 1.7 GB | Complete trained model (10 epochs) | [HuggingFace](#fully-trained-checkpoint) |
| **FinBERT** | 438 MB | Fine-tuned FinBERT for NER | [HuggingFace](#finbert-checkpoint) |
| **Stage Checkpoints** | 1.6 GB each | Individual stage checkpoints | [HuggingFace](#stage-checkpoints) |

**Total Download:** ~2.1 GB (fully trained model only) or ~5.3 GB (all checkpoints)

---

## 🚀 Quick Start

### Option 1: Download Fully Trained Model (Recommended)

```bash
# Install Hugging Face Hub
pip install huggingface_hub

# Download fully trained model
python scripts/download_checkpoints.py --model fully_trained

# Or manually:
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="your-username/financial-explainable-summarization",
    filename="fully_trained_model.pt",
    cache_dir="checkpoints/"
)
```

### Option 2: Download All Checkpoints

```bash
# Download all checkpoints
python scripts/download_checkpoints.py --all

# Total size: ~5.3 GB
```

---

## 📥 Checkpoint Details

### 1. Base Model Checkpoint

**Purpose:** Starting point for fine-tuning or transfer learning

**Filename:** `base_model.pt`  
**Size:** 1.6 GB  
**Components:**
- BART-large encoder-decoder (406M parameters)
- Initialized KG encoder (2.1M parameters)
- Initialized cross-attention (4.7M parameters)
- Initialized multi-hop reasoning (3.2M parameters)

**HuggingFace Download:**
```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="your-username/financial-explainable-summarization",
    filename="base_model.pt",
    cache_dir="checkpoints/"
)
```

**Direct URL:**
```
https://huggingface.co/your-username/financial-explainable-summarization/resolve/main/base_model.pt
```

**Load in Code:**
```python
import torch
from models import HybridModel

# Initialize model
model = HybridModel.from_pretrained("checkpoints/base_model.pt")

# Or load checkpoint manually
checkpoint = torch.load("checkpoints/base_model.pt")
model = HybridModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
```

**Use Cases:**
- Transfer learning to new domains
- Fine-tuning with custom data
- Experimenting with different training strategies

---

### 2. Fully Trained Model Checkpoint ⭐ **RECOMMENDED**

**Purpose:** Production-ready model for inference and evaluation

**Filename:** `fully_trained_model.pt`  
**Size:** 1.7 GB  
**Training Details:**
- 10 epochs (3 + 2 + 5 across 3 stages)
- Training time: ~28 hours on A100 40GB
- Final validation loss: 1.23

**Performance (Table II):**
- ROUGE-L: 0.487
- BERTScore: 0.686
- Factual Consistency: 87.6%

**Explainability (Table III):**
- SSI: 0.74 ± 0.06
- CPS: 0.51 ± 0.07
- TCC: 0.54 ± 0.07
- Consistency: 0.59 ± 0.06

**HuggingFace Download:**
```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="your-username/financial-explainable-summarization",
    filename="fully_trained_model.pt",
    cache_dir="checkpoints/",
    resume_download=True  # Resume if interrupted
)
```

**Alternative: Google Drive**
```bash
# Install gdown
pip install gdown

# Download from Google Drive
gdown https://drive.google.com/uc?id=YOUR_FILE_ID \
    -O checkpoints/fully_trained_model.pt
```

**Alternative: Direct Download**
```bash
# Using wget
wget https://huggingface.co/your-username/financial-explainable-summarization/resolve/main/fully_trained_model.pt \
    -O checkpoints/fully_trained_model.pt

# Using curl
curl -L https://huggingface.co/your-username/financial-explainable-summarization/resolve/main/fully_trained_model.pt \
    -o checkpoints/fully_trained_model.pt
```

**Load and Use:**
```python
import torch
from models import HybridModel
from transformers import AutoTokenizer

# Load model
model = HybridModel.from_pretrained("checkpoints/fully_trained_model.pt")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

# Generate summary
text = "Apple Inc. reported Q4 earnings of $89.5B..."
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

**Checkpoint Contents:**
```python
checkpoint = torch.load("checkpoints/fully_trained_model.pt")

# Available keys:
checkpoint.keys()
# dict_keys(['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
#            'epoch', 'stage', 'config', 'training_history', 'metrics'])

# Model weights
model.load_state_dict(checkpoint['model_state_dict'])

# Training history
history = checkpoint['training_history']
print(f"Train loss: {history['train_loss']}")
print(f"Val loss: {history['val_loss']}")

# Evaluation metrics
metrics = checkpoint['metrics']
print(f"ROUGE-L: {metrics['rouge_l']}")
print(f"BERTScore: {metrics['bertscore']}")
```

---

### 3. FinBERT Checkpoint (Custom Fine-tuned)

**Purpose:** Financial NER and entity recognition

**Filename:** `finbert_ner.pt`  
**Size:** 438 MB  
**Base Model:** ProsusAI/finbert  
**Fine-tuning:** Financial entity recognition on 10K corpus

**HuggingFace Download:**
```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="your-username/financial-explainable-summarization",
    filename="finbert_ner.pt",
    cache_dir="checkpoints/"
)
```

**Load and Use:**
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load FinBERT NER model
model = AutoModelForTokenClassification.from_pretrained(
    "checkpoints/finbert_ner.pt"
)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# Extract entities
text = "Apple Inc. CEO Tim Cook announced revenue of $89.5B."
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Decode entities
entities = []
for idx, pred in enumerate(predictions[0]):
    if pred != 0:  # Not 'O' tag
        token = tokenizer.decode(inputs['input_ids'][0][idx])
        label = model.config.id2label[pred.item()]
        entities.append((token, label))

print(entities)
# [('Apple', 'ORG'), ('Inc', 'ORG'), ('Tim', 'PERSON'), ('Cook', 'PERSON'), ...]
```

**Entity Types:**
- ORG: Organizations, companies
- PERSON: People, executives
- PRODUCT: Financial products
- MONEY: Monetary amounts
- PERCENT: Percentages
- DATE: Dates and periods
- GPE: Geopolitical entities

---

### 4. Stage Checkpoints

**Purpose:** Individual checkpoints from each training stage

**Available Stages:**

#### **Stage 1: Warm-up (After 3 epochs)**
```python
# Download
checkpoint_path = hf_hub_download(
    repo_id="your-username/financial-explainable-summarization",
    filename="stage1_final.pt",
    cache_dir="checkpoints/"
)

# Load
checkpoint = torch.load("checkpoints/stage1_final.pt")
# Trained components: decoder only
# Validation loss: 2.45
```

#### **Stage 2: KG Integration (After 5 epochs total)**
```python
# Download
checkpoint_path = hf_hub_download(
    repo_id="your-username/financial-explainable-summarization",
    filename="stage2_final.pt",
    cache_dir="checkpoints/"
)

# Load
checkpoint = torch.load("checkpoints/stage2_final.pt")
# Trained components: text_encoder, kg_encoder, cross_attention
# Validation loss: 1.67
```

#### **Stage 3: End-to-End (After 10 epochs total)**
```python
# Download
checkpoint_path = hf_hub_download(
    repo_id="your-username/financial-explainable-summarization",
    filename="stage3_final.pt",
    cache_dir="checkpoints/"
)

# This is the same as fully_trained_model.pt
```

---

## 🔧 Download Script

Create `scripts/download_checkpoints.py`:

```python
#!/usr/bin/env python3
"""
Download Model Checkpoints
==========================

Download pre-trained model checkpoints from HuggingFace Hub.

Usage:
    python scripts/download_checkpoints.py --model fully_trained
    python scripts/download_checkpoints.py --all
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Configuration
REPO_ID = "your-username/financial-explainable-summarization"
CACHE_DIR = "checkpoints/"

CHECKPOINTS = {
    'base': {
        'filename': 'base_model.pt',
        'size': '1.6 GB',
        'description': 'Base model with initialized components'
    },
    'fully_trained': {
        'filename': 'fully_trained_model.pt',
        'size': '1.7 GB',
        'description': 'Complete trained model (recommended)'
    },
    'finbert': {
        'filename': 'finbert_ner.pt',
        'size': '438 MB',
        'description': 'Fine-tuned FinBERT for NER'
    },
    'stage1': {
        'filename': 'stage1_final.pt',
        'size': '1.6 GB',
        'description': 'Stage 1 checkpoint (warm-up)'
    },
    'stage2': {
        'filename': 'stage2_final.pt',
        'size': '1.6 GB',
        'description': 'Stage 2 checkpoint (KG integration)'
    },
    'stage3': {
        'filename': 'stage3_final.pt',
        'size': '1.7 GB',
        'description': 'Stage 3 checkpoint (end-to-end)'
    }
}


def download_checkpoint(checkpoint_name: str, cache_dir: str = CACHE_DIR):
    """Download a specific checkpoint"""
    if checkpoint_name not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}")
    
    info = CHECKPOINTS[checkpoint_name]
    print(f"\n{'='*70}")
    print(f"Downloading: {checkpoint_name}")
    print(f"Description: {info['description']}")
    print(f"Size: {info['size']}")
    print(f"{'='*70}\n")
    
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Download from HuggingFace
    try:
        checkpoint_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=info['filename'],
            cache_dir=cache_dir,
            resume_download=True
        )
        
        print(f"\n✓ Downloaded successfully!")
        print(f"Location: {checkpoint_path}")
        return checkpoint_path
    
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print(f"\nAlternative download methods:")
        print(f"1. Manual download from:")
        print(f"   https://huggingface.co/{REPO_ID}/resolve/main/{info['filename']}")
        print(f"2. Google Drive (if available)")
        print(f"3. Contact authors for direct access")
        return None


def download_all(cache_dir: str = CACHE_DIR):
    """Download all checkpoints"""
    print(f"\n{'='*70}")
    print("Downloading All Checkpoints")
    print(f"Total size: ~5.3 GB")
    print(f"{'='*70}")
    
    for name in CHECKPOINTS.keys():
        download_checkpoint(name, cache_dir)
    
    print(f"\n{'='*70}")
    print("All downloads complete!")
    print(f"{'='*70}")


def verify_checkpoint(checkpoint_path: str):
    """Verify checkpoint integrity"""
    import torch
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n{'='*70}")
        print("Checkpoint Verification")
        print(f"{'='*70}")
        
        print(f"\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        if 'config' in checkpoint:
            print(f"\nModel configuration:")
            print(f"  Hidden dim: {checkpoint['config'].get('hidden_dim', 'N/A')}")
            print(f"  Num layers: {checkpoint['config'].get('num_layers', 'N/A')}")
        
        if 'metrics' in checkpoint:
            print(f"\nPerformance metrics:")
            for metric, value in checkpoint['metrics'].items():
                print(f"  {metric}: {value}")
        
        print(f"\n✓ Checkpoint verified successfully!")
        return True
    
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained model checkpoints"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['base', 'fully_trained', 'finbert', 'stage1', 'stage2', 'stage3'],
        help='Specific checkpoint to download'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all checkpoints'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=CACHE_DIR,
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify checkpoint after download'
    )
    
    args = parser.parse_args()
    
    if args.all:
        download_all(args.cache_dir)
    elif args.model:
        checkpoint_path = download_checkpoint(args.model, args.cache_dir)
        
        if checkpoint_path and args.verify:
            verify_checkpoint(checkpoint_path)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/download_checkpoints.py --model fully_trained")
        print("  python scripts/download_checkpoints.py --all")


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Download fully trained model
python scripts/download_checkpoints.py --model fully_trained

# Download all checkpoints
python scripts/download_checkpoints.py --all

# Download and verify
python scripts/download_checkpoints.py --model fully_trained --verify
```

---

## 📊 Checkpoint Comparison

| Checkpoint | Trainable Params | Frozen Params | Val Loss | ROUGE-L | Use Case |
|------------|-----------------|---------------|----------|---------|----------|
| **Base** | 0 | 416M | N/A | N/A | Fine-tuning |
| **Stage 1** | 12M (decoder) | 404M | 2.45 | 0.38 | Warm-up analysis |
| **Stage 2** | 410M | 6M | 1.67 | 0.45 | KG integration study |
| **Stage 3** | 416M | 0 | 1.23 | 0.487 | Production use ⭐ |
| **FinBERT** | 110M | 0 | N/A | N/A | Entity extraction |

---

## 🔐 Checkpoint Verification

### Verify Integrity

```python
import torch
import hashlib

def verify_checkpoint(checkpoint_path: str, expected_hash: str = None):
    """Verify checkpoint integrity"""
    
    # Compute SHA256 hash
    sha256 = hashlib.sha256()
    
    with open(checkpoint_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    
    file_hash = sha256.hexdigest()
    
    print(f"File: {checkpoint_path}")
    print(f"SHA256: {file_hash}")
    
    if expected_hash:
        if file_hash == expected_hash:
            print("✓ Verification successful!")
            return True
        else:
            print("✗ Hash mismatch! File may be corrupted.")
            return False
    
    return True

# Verify
verify_checkpoint("checkpoints/fully_trained_model.pt")
```

### Expected Hashes (SHA256)

```yaml
# Update after uploading to HuggingFace
checksums:
  base_model.pt: "abc123..."
  fully_trained_model.pt: "def456..."
  finbert_ner.pt: "ghi789..."
  stage1_final.pt: "jkl012..."
  stage2_final.pt: "mno345..."
  stage3_final.pt: "pqr678..."
```

---

## 💾 Storage Locations

### Primary: HuggingFace Hub (Recommended)

**Repository:**
```
https://huggingface.co/your-username/financial-explainable-summarization
```

**Files:**
- `base_model.pt`
- `fully_trained_model.pt`
- `finbert_ner.pt`
- `stage1_final.pt`
- `stage2_final.pt`
- `stage3_final.pt`

**Advantages:**
- ✅ Free unlimited storage for ML models
- ✅ Version control with Git LFS
- ✅ Easy integration with transformers
- ✅ Built-in download resumption
- ✅ CDN distribution

### Alternative: Google Drive

**Shared Folder:**
```
https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

**Download with gdown:**
```bash
pip install gdown

# Download fully trained model
gdown https://drive.google.com/uc?id=YOUR_FILE_ID \
    -O checkpoints/fully_trained_model.pt
```

### Alternative: OneDrive/Dropbox

**OneDrive:**
```bash
# Generate direct download link from OneDrive sharing URL
# Replace with your actual link
wget 'https://onedrive.live.com/download?...' \
    -O checkpoints/fully_trained_model.pt
```

---

## 🚀 Usage Examples

### Example 1: Quick Inference

```python
from fin_explainable import load_model, summarize

# Load fully trained model
model, tokenizer = load_model("checkpoints/fully_trained_model.pt")

# Summarize document
document = """
Apple Inc. reported quarterly earnings of $89.5 billion for Q4 2023,
representing a 12% year-over-year increase. CEO Tim Cook attributed
the growth to strong iPhone sales and services revenue...
"""

summary = summarize(model, tokenizer, document)
print(summary)
```

### Example 2: Evaluation

```python
from evaluation import EvaluationPipeline

# Load model
model = load_model("checkpoints/fully_trained_model.pt")

# Run evaluation
pipeline = EvaluationPipeline(model, tokenizer)
results = pipeline.evaluate(
    test_dataloader=test_loader,
    output_dir="evaluation_results/"
)

print(f"ROUGE-L: {results['rouge_l']:.3f}")
print(f"BERTScore: {results['bertscore']:.3f}")
```

### Example 3: Fine-tuning from Base

```python
from training import MultiStageTrainer

# Load base model
model = load_model("checkpoints/base_model.pt")

# Fine-tune on custom data
trainer = MultiStageTrainer(model, train_loader, val_loader, config)
trainer.train(output_dir="custom_checkpoints/")
```

---

## 📝 Checkpoint Metadata

Each checkpoint includes:

```python
checkpoint = {
    'model_state_dict': {...},      # Model weights
    'optimizer_state_dict': {...},  # Optimizer state (if saved)
    'scheduler_state_dict': {...},  # Scheduler state (if saved)
    'epoch': 10,                     # Training epoch
    'stage': 3,                      # Training stage
    'config': {...},                 # Model configuration
    'training_history': {            # Training curves
        'train_loss': [...],
        'val_loss': [...]
    },
    'metrics': {                     # Evaluation metrics
        'rouge_l': 0.487,
        'bertscore': 0.686,
        'factual_consistency': 87.6
    },
    'timestamp': '2024-12-22T10:30:00',
    'git_commit': 'abc123...',
    'pytorch_version': '2.0.1',
    'cuda_version': '11.8'
}
```

---

## ⚠️ Troubleshooting

### Download Issues

**Problem:** Download is slow
```bash
# Solution 1: Use mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com

# Solution 2: Use aria2c for faster downloads
pip install aria2c
aria2c -x 16 -s 16 https://huggingface.co/.../model.pt
```

**Problem:** Download interrupted
```bash
# Solution: Downloads automatically resume
python scripts/download_checkpoints.py --model fully_trained
# Will resume from where it stopped
```

**Problem:** Out of disk space
```bash
# Check space
df -h checkpoints/

# Solution: Download only needed checkpoint
python scripts/download_checkpoints.py --model fully_trained
# Don't use --all if space is limited
```

### Loading Issues

**Problem:** CUDA out of memory
```python
# Solution: Load on CPU first
checkpoint = torch.load(
    "checkpoints/fully_trained_model.pt",
    map_location='cpu'
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cuda')  # Then move to GPU
```

**Problem:** Version mismatch
```python
# Solution: Update PyTorch
pip install torch==2.0.1 --upgrade

# Or use weights_only for safety
checkpoint = torch.load(
    "checkpoints/fully_trained_model.pt",
    weights_only=True
)
```

---

## 📧 Support

**Issues downloading or using checkpoints?**

1. Check [GitHub Issues](https://github.com/your-username/financial-explainable-summarization/issues)
2. Contact authors:
   - Sumeer Riaz: sumeer33885@iqraisb.edu.pk
   - Dr. M. Bilal Bashir: bilal.bashir@iqraisb.edu.pk
3. HuggingFace discussions: [Link to repo discussions]

---

## 📄 License

All checkpoints are released under the same MIT License as the source code.

**Citation:**
```bibtex
@article{riaz2024explainable,
  title={An eXplainable Approach to Abstractive Text Summarization Using External Knowledge},
  author={Riaz, Sumeer and Bashir, M. Bilal and Naqvi, Syed Ali Hassan},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

---

## ✅ Checklist

Before using checkpoints:

- [ ] Downloaded required checkpoint(s)
- [ ] Verified file integrity (optional)
- [ ] Tested loading in code
- [ ] Checked GPU memory requirements
- [ ] Read model card and limitations
- [ ] Cited paper if using in research

---

## 🎯 Quick Reference

**Download fully trained model:**
```bash
python scripts/download_checkpoints.py --model fully_trained
```

**Load and use:**
```python
from fin_explainable import load_model, summarize

model, tokenizer = load_model("checkpoints/fully_trained_model.pt")
summary = summarize(model, tokenizer, "Your text here...")
```

**Expected performance:**
- ROUGE-L: 0.487
- BERTScore: 0.686
- Inference: 2.9s/document

---

**Ready to use pre-trained models! 🚀**
