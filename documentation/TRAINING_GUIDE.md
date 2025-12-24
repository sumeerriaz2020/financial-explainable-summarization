# Training Guide

Step-by-step instructions for training the hybrid neural-symbolic model.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Data Preparation](#data-preparation)
- [Multi-Stage Training](#multi-stage-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software Requirements

```bash
# Python 3.8+
python --version

# CUDA 11.8+ (for GPU training)
nvcc --version

# Install dependencies
conda env create -f environment.yml
conda activate fin-explainable
pip install -e .
```

### Download FIBO Ontology

```bash
# Create data directory
mkdir -p data/fibo

# Download FIBO 2024-Q1
wget https://spec.edmcouncil.org/fibo/ontology/master/2024Q1/AboutFIBOProd.rdf \
    -O data/fibo/fibo_2024_q1.owl
```

---

## System Requirements

### Minimum Requirements (Training Possible)

- **GPU:** 1x NVIDIA A100 (40GB) or 2x V100 (32GB)
- **RAM:** 64 GB
- **Storage:** 100 GB free space
- **Training Time:** ~30 hours

### Recommended Requirements (Optimal Performance)

- **GPU:** 4x NVIDIA A100 (40GB)
- **RAM:** 256 GB
- **Storage:** 500 GB SSD
- **Training Time:** ~8 hours (with distributed training)

### Budget Estimate

| Configuration | Cost | Training Time |
|---------------|------|---------------|
| **Single A100 (40GB)** | ~$75 | 30 hours |
| **4x A100 (40GB)** | ~$225 | 8 hours |
| **Spot Instances** | ~$50-150 | 8-30 hours |

**Cloud Platform Pricing (per hour):**
- AWS p4d.24xlarge (8x A100): ~$32/hour
- GCP a2-ultragpu-8g (8x A100): ~$28/hour
- Azure ND96amsr_A100_v4 (8x A100): ~$27/hour

**Total Training Cost:**
- With 4x A100 for 8 hours: **~$225**
- With 1x A100 for 30 hours: **~$75**

---

## Data Preparation

### 1. Prepare Training Data

**Expected Format:**
```json
{
  "documents": [
    {
      "id": "doc_001",
      "text": "Full financial document text...",
      "summary": "Reference summary...",
      "metadata": {
        "source": "10-K filing",
        "company": "Apple Inc.",
        "date": "2023-Q4"
      }
    }
  ]
}
```

**Create Dataset:**
```python
# scripts/prepare_data.py
import json
from pathlib import Path

def prepare_dataset(raw_data_path: str, output_path: str):
    """Prepare training dataset"""
    
    # Load raw data
    with open(raw_data_path) as f:
        raw_data = json.load(f)
    
    # Process documents
    processed = []
    for item in raw_data:
        processed.append({
            'id': item['id'],
            'text': preprocess_text(item['text']),
            'summary': preprocess_text(item['summary']),
            'metadata': item.get('metadata', {})
        })
    
    # Save
    with open(output_path, 'w') as f:
        json.dump({'documents': processed}, f, indent=2)
    
    print(f"Processed {len(processed)} documents")
    print(f"Saved to: {output_path}")

# Run
prepare_dataset('data/raw/financial_corpus.json', 'data/train.json')
```

### 2. Split Data

```python
from sklearn.model_selection import train_test_split
import json

# Load data
with open('data/train.json') as f:
    data = json.load(f)['documents']

# Split: 80% train, 10% val, 10% test
train, temp = train_test_split(data, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save splits
for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
    with open(f'data/{split_name}.json', 'w') as f:
        json.dump({'documents': split_data}, f, indent=2)
    print(f"{split_name}: {len(split_data)} documents")
```

**Expected Output:**
```
train: 800 documents
val: 100 documents  
test: 100 documents
```

### 3. Construct Knowledge Graphs

```python
from knowledge_graph import FIBOIntegration, KnowledgeGraphConstructor
import json

# Initialize
fibo = FIBOIntegration()
kg_constructor = KnowledgeGraphConstructor(fibo)

# Load training data
with open('data/train.json') as f:
    documents = [d['text'] for d in json.load(f)['documents']]

# Construct knowledge graphs
print("Constructing knowledge graphs...")
for i, doc in enumerate(documents):
    kg = kg_constructor.construct_knowledge_graph([doc])
    
    # Save KG
    kg.save(f'data/knowledge_graphs/kg_{i:05d}.pkl')
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(documents)} documents")

print("Knowledge graph construction complete!")
```

---

## Multi-Stage Training

### Stage 1: Warm-up Phase (3 epochs)

**Purpose:** Initialize decoder, freeze encoders

**Configuration:**
```yaml
# configs/stage1_config.yaml
stage1:
  epochs: 3
  learning_rate: 1.0e-4
  batch_size: 16
  
  trainable_components:
    - decoder
  
  frozen_components:
    - text_encoder.embeddings
    - kg_encoder
    - cross_attention
  
  loss_weights:
    summary: 1.0
    kg_alignment: 0.0
    explanation: 0.0
```

**Training Script:**
```python
from training import MultiStageTrainer
from models import HybridModel
from torch.utils.data import DataLoader
import yaml

# Load config
with open('configs/stage1_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize model
model = HybridModel()

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Initialize trainer
trainer = MultiStageTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=config,
    device='cuda'
)

# Train Stage 1
print("=" * 70)
print("STAGE 1: Warm-up Phase")
print("=" * 70)
trainer.train_stage(stage=1, output_dir='checkpoints/stage1/')
```

**Expected Output:**
```
STAGE 1: Warm-up Phase
======================================================================
Epoch 1/3
Train Loss: 2.45 | Val Loss: 2.67 | Time: 2.5h
Epoch 2/3
Train Loss: 2.10 | Val Loss: 2.38 | Time: 2.5h
Epoch 3/3
Train Loss: 1.98 | Val Loss: 2.25 | Time: 2.5h

Stage 1 Complete! Total time: 7.5 hours
Best checkpoint saved: checkpoints/stage1/best_model.pt
```

---

### Stage 2: Knowledge Integration (2 epochs)

**Purpose:** Integrate KG encoder and cross-attention

**Configuration:**
```yaml
# configs/stage2_config.yaml
stage2:
  epochs: 2
  learning_rate: 5.0e-5
  batch_size: 16
  
  trainable_components:
    - text_encoder
    - kg_encoder
    - cross_attention
  
  frozen_components:
    - decoder
  
  loss_weights:
    summary: 0.7
    kg_alignment: 0.3
    explanation: 0.0
```

**Training:**
```python
# Load Stage 1 checkpoint
checkpoint = torch.load('checkpoints/stage1/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Train Stage 2
print("=" * 70)
print("STAGE 2: Knowledge Integration")
print("=" * 70)
trainer.train_stage(stage=2, output_dir='checkpoints/stage2/')
```

**Expected Output:**
```
STAGE 2: Knowledge Integration
======================================================================
Epoch 1/2
Train Loss: 1.75 | Val Loss: 1.82 | KG Alignment: 0.45 | Time: 3h
Epoch 2/2
Train Loss: 1.65 | Val Loss: 1.70 | KG Alignment: 0.38 | Time: 3h

Stage 2 Complete! Total time: 6 hours
Best checkpoint saved: checkpoints/stage2/best_model.pt
```

---

### Stage 3: End-to-End Fine-tuning (5 epochs)

**Purpose:** Fine-tune all components together

**Configuration:**
```yaml
# configs/stage3_config.yaml
stage3:
  epochs: 5
  learning_rate: 2.0e-5
  batch_size: 16
  
  trainable_components:
    - all
  
  loss_weights:
    summary: 0.7    # α
    kg_alignment: 0.2  # β
    explanation: 0.1   # γ
```

**Training:**
```python
# Load Stage 2 checkpoint
checkpoint = torch.load('checkpoints/stage2/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Train Stage 3
print("=" * 70)
print("STAGE 3: End-to-End Fine-tuning")
print("=" * 70)
trainer.train_stage(stage=3, output_dir='checkpoints/stage3/')
```

**Expected Output:**
```
STAGE 3: End-to-End Fine-tuning
======================================================================
Epoch 1/5
Train Loss: 1.45 | Val Loss: 1.52 | Time: 2.5h
Epoch 2/5
Train Loss: 1.35 | Val Loss: 1.42 | Time: 2.5h
Epoch 3/5
Train Loss: 1.28 | Val Loss: 1.35 | Time: 2.5h
Epoch 4/5
Train Loss: 1.24 | Val Loss: 1.30 | Time: 2.5h
Epoch 5/5
Train Loss: 1.21 | Val Loss: 1.27 | Time: 2.5h

Stage 3 Complete! Total time: 12.5 hours
Best checkpoint saved: checkpoints/stage3/best_model.pt
Final model saved: checkpoints/fully_trained_model.pt
```

---

### Complete Training Pipeline

**Single Command Training:**
```bash
python train.py \
    --config configs/training_config.yaml \
    --data_dir data/ \
    --output_dir checkpoints/ \
    --num_gpus 4 \
    --distributed
```

**Or use Python:**
```python
from training import MultiStageTrainer

# One-line training
trainer = MultiStageTrainer(model, train_loader, val_loader, config)
trainer.train('checkpoints/')  # Runs all 3 stages automatically
```

**Total Training Time:**
- Stage 1: ~8 hours (3 epochs)
- Stage 2: ~6 hours (2 epochs)
- Stage 3: ~12 hours (5 epochs)
- **Total: ~26 hours** (single A100)
- **Total: ~8 hours** (4x A100 distributed)

---

## Hyperparameter Tuning

### Bayesian Optimization

```python
from training import HyperparameterOptimizer

def train_fn(model, config, max_epochs=5):
    """Quick training for HP search"""
    trainer = MultiStageTrainer(model, train_loader, val_loader, config)
    history = trainer.train_quick(max_epochs=max_epochs)
    return history

def eval_fn(model):
    """Evaluation for HP search"""
    from evaluation import EvaluationPipeline
    pipeline = EvaluationPipeline(model, tokenizer)
    results = pipeline.evaluate(val_loader, 'hp_eval/')
    return results['val_loss']

# Run optimization
optimizer = HyperparameterOptimizer(
    model_class=HybridModel,
    train_fn=train_fn,
    eval_fn=eval_fn,
    n_trials=50
)

best_params = optimizer.optimize()

print("Best hyperparameters:")
print(best_params)
```

**Expected Runtime:** ~8-12 hours (50 trials)

**Optimized Parameters:**
```python
{
    'stage1_lr': 8.3e-05,
    'stage2_lr': 3.2e-05,
    'stage3_lr': 1.5e-05,
    'alpha': 0.72,
    'beta': 0.18,
    'gamma': 0.10,
    'hidden_dim': 768,
    'num_attention_heads': 8,
    'num_reasoning_hops': 3,
    'dropout': 0.15
}
```

---

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/

# View at http://localhost:6006
```

**Tracked Metrics:**
- Train/Val Loss
- ROUGE-L
- BERTScore
- Learning Rate
- Gradient Norm

### Weights & Biases (Optional)

```python
import wandb

wandb.init(
    project="financial-summarization",
    config=config
)

# Logs automatically during training
```

### Training Logs

```bash
# View live logs
tail -f logs/training.log

# Grep for errors
grep "ERROR" logs/training.log
```

---

## Distributed Training

### 4 GPU Training

```python
# Use DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = HybridModel()
model = DDP(model, device_ids=[local_rank])

# Train
trainer = MultiStageTrainer(model, train_loader, val_loader, config)
trainer.train('checkpoints/')
```

**Launch Script:**
```bash
# Using torchrun
torchrun --nproc_per_node=4 train.py \
    --config configs/training_config.yaml \
    --distributed

# Or use launch.py
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py --distributed
```

**Speedup:**
- 1 GPU: 30 hours
- 2 GPUs: 16 hours (1.9x)
- 4 GPUs: 8 hours (3.8x)

---

## Checkpointing

### Save Checkpoints

```python
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
```

### Resume Training

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/stage2_epoch_1.pt')

# Restore state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training
for epoch in range(start_epoch, total_epochs):
    train_one_epoch(...)
```

---

## Troubleshooting

### Out of Memory (OOM)

**Problem:** CUDA out of memory

**Solutions:**
```python
# 1. Reduce batch size
config['batch_size'] = 8  # instead of 16

# 2. Enable gradient accumulation
config['gradient_accumulation_steps'] = 2

# 3. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(...)
```

### Slow Training

**Problem:** Training is very slow

**Solutions:**
```python
# 1. Enable pin_memory
train_loader = DataLoader(
    dataset,
    batch_size=16,
    pin_memory=True,
    num_workers=4
)

# 2. Use faster data loading
from torch.utils.data import DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True
)

# 3. Profile code
with torch.profiler.profile() as prof:
    train_one_epoch(...)
print(prof.key_averages().table())
```

### Loss Not Decreasing

**Problem:** Validation loss stuck or increasing

**Solutions:**
```python
# 1. Reduce learning rate
config['learning_rate'] = 1e-5  # instead of 2e-5

# 2. Add warmup
config['warmup_steps'] = 500

# 3. Check gradient clipping
config['max_grad_norm'] = 1.0

# 4. Verify data quality
print(f"Train samples: {len(train_dataset)}")
print(f"Sample text length: {len(train_dataset[0]['text'])}")
```

### NaN Loss

**Problem:** Loss becomes NaN during training

**Solutions:**
```python
# 1. Check for NaN in data
assert not torch.isnan(inputs).any()

# 2. Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Reduce learning rate
config['learning_rate'] = 1e-6

# 4. Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
```

---

## Expected Results

After successful training:

**Summarization Quality (Table II):**
- ROUGE-L: 0.487 (±0.02)
- BERTScore: 0.686 (±0.01)
- Factual Consistency: 87.6% (±2%)

**Explainability (Table III):**
- SSI: 0.74 ± 0.06
- CPS: 0.51 ± 0.07
- TCC: 0.54 ± 0.07

**Training Time:**
- Single A100: ~30 hours
- 4x A100: ~8 hours

**Cost:**
- Single A100: ~$75
- 4x A100: ~$225

---

## Next Steps

After training:
1. Run evaluation: `python evaluate.py --checkpoint checkpoints/fully_trained_model.pt`
2. Generate predictions: `python inference.py --model checkpoints/fully_trained_model.pt`
3. Upload to HuggingFace: `huggingface-cli upload ...`
4. Deploy model (see DEPLOYMENT_GUIDE.md)

---

For questions, see [Troubleshooting](#troubleshooting) or contact authors.
