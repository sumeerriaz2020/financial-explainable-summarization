# Reproducibility Guide

Complete instructions for reproducing results from the paper.

---

## Fixed Random Seeds

All experiments use **seed=42** for reproducibility.

```python
import torch
import numpy as np
import random

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional settings
    torch.use_deterministic_algorithms(True)

# Apply
set_seed(42)
```

---

## Exact Environment

### Software Versions

```yaml
python: 3.10
pytorch: 2.0.1
transformers: 4.35.0
cuda: 11.8
cudnn: 8.7
```

### Hardware Configuration

**Paper Results Used:**
- GPU: 4x NVIDIA A100 (40GB)
- CPU: AMD EPYC 7763 64-Core
- RAM: 256 GB
- Storage: 1 TB NVMe SSD

**Minimum for Reproduction:**
- GPU: 1x NVIDIA A100 (40GB) or 2x V100 (32GB)
- CPU: 16+ cores
- RAM: 64 GB
- Storage: 100 GB

---

## Data Preparation

### Dataset Statistics (Paper)

```
Training set: 800 documents
Validation set: 100 documents
Test set: 100 documents

Avg document length: 1,245 tokens
Avg summary length: 87 tokens
Domain: Financial (10-K, earnings reports)
```

### Data Split

```python
from sklearn.model_selection import train_test_split

# Load data
data = load_financial_corpus()

# Split with fixed seed
train, temp = train_test_split(data, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save
save_split('data/train.json', train)
save_split('data/val.json', val)
save_split('data/test.json', test)
```

---

## Training Configuration

### Exact Hyperparameters

```yaml
# Stage 1 (Warm-up)
epochs: 3
learning_rate: 1.0e-4
batch_size: 16
optimizer: AdamW
weight_decay: 0.01
warmup_ratio: 0.1
max_grad_norm: 1.0

# Stage 2 (KG Integration)
epochs: 2
learning_rate: 5.0e-5
batch_size: 16

# Stage 3 (End-to-End)
epochs: 5
learning_rate: 2.0e-5
batch_size: 16

# Loss weights (Equation 10)
alpha: 0.7   # Summary loss
beta: 0.2    # KG alignment
gamma: 0.1   # Explanation loss
```

### Model Architecture

```python
model_config = {
    'text_encoder': 'facebook/bart-large',
    'hidden_dim': 1024,
    'kg_encoder': {
        'input_dim': 768,
        'hidden_dim': 512,
        'output_dim': 1024,
        'num_layers': 3,
        'num_heads': 8
    },
    'cross_attention': {
        'num_heads': 8,
        'dropout': 0.1
    },
    'num_reasoning_hops': 3,
    'dropout': 0.1
}
```

---

## Training Procedure

### Step-by-Step

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate fin-explainable

# 2. Set seed
export PYTHONHASHSEED=42

# 3. Download FIBO
python scripts/download_fibo.sh

# 4. Prepare data
python scripts/prepare_data.py --seed 42

# 5. Train model
python train.py \
    --config configs/training_config.yaml \
    --seed 42 \
    --deterministic
```

### Training Time

- Stage 1: 7.8 hours
- Stage 2: 5.9 hours
- Stage 3: 12.3 hours
- **Total: 26.0 hours** (single A100)

---

## Evaluation

### Reproduce Table II Results

```bash
python evaluate.py \
    --checkpoint checkpoints/fully_trained_model.pt \
    --test_data data/test.json \
    --seed 42 \
    --output results/table_ii.json
```

**Expected Output:**
```json
{
  "rouge_l": 0.487,
  "rouge_1": 0.512,
  "rouge_2": 0.398,
  "bertscore": 0.686,
  "factual_consistency": 87.6
}
```

### Reproduce Table III Results

```bash
python evaluate_explainability.py \
    --checkpoint checkpoints/fully_trained_model.pt \
    --test_data data/test.json \
    --seed 42 \
    --output results/table_iii.json
```

**Expected Output:**
```json
{
  "ssi": 0.74,
  "ssi_std": 0.06,
  "cps": 0.51,
  "cps_std": 0.07,
  "tcc": 0.54,
  "tcc_std": 0.07,
  "consistency": 0.59,
  "consistency_std": 0.06
}
```

---

## Error Analysis

### Reproduce Section V.D

```bash
python error_analysis.py \
    --checkpoint checkpoints/fully_trained_model.pt \
    --test_data data/test.json \
    --output results/error_analysis.json
```

**Expected Distribution:**
```
Entity Misidentification: 21.8%
Causal Misattribution: 15.2%
Temporal Inconsistency: 12.5%
Factual Errors: 7.0%
Other: 43.5%
Total Error Rate: 56.5%
```

---

## Checksum Verification

### Model Checksums

```bash
# Verify downloaded checkpoint
sha256sum checkpoints/fully_trained_model.pt

# Expected (update after upload):
# abc123def456... checkpoints/fully_trained_model.pt
```

### Data Checksums

```bash
sha256sum data/train.json data/val.json data/test.json

# Expected:
# Train: xyz789...
# Val: abc456...
# Test: def123...
```

---

## Common Reproducibility Issues

### Issue 1: Different Results with Same Seed

**Cause:** Non-deterministic CUDA operations

**Solution:**
```python
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

### Issue 2: Batch Size Affects Results

**Cause:** Gradient accumulation differences

**Solution:** Use exact batch size (16) or adjust accumulation:
```python
effective_batch = 16
actual_batch = 8
accumulation_steps = effective_batch // actual_batch  # = 2
```

### Issue 3: Hardware Differences

**Cause:** Different GPU architectures

**Impact:** ±0.5% variance in metrics

**Solution:** Use same GPU family (A100 recommended)

---

## Variance Analysis

### Expected Variance

Running the same configuration 5 times:

| Metric | Mean | Std Dev | Range |
|--------|------|---------|-------|
| ROUGE-L | 0.487 | ±0.005 | 0.482-0.492 |
| BERTScore | 0.686 | ±0.003 | 0.683-0.689 |
| Factual | 87.6% | ±0.8% | 86.8-88.4% |
| SSI | 0.74 | ±0.02 | 0.72-0.76 |

**Conclusion:** Results stable within ±1%

---

## Reproducibility Checklist

Before claiming reproduction:

- [ ] Exact Python version (3.10)
- [ ] Exact PyTorch version (2.0.1)
- [ ] Exact CUDA version (11.8)
- [ ] Seed set to 42
- [ ] Deterministic mode enabled
- [ ] Same data split
- [ ] Same hyperparameters
- [ ] Same training procedure
- [ ] Same evaluation protocol
- [ ] Results within ±1% variance

---

## Docker for Perfect Reproducibility

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set seeds
ENV PYTHONHASHSEED=42

# Install exact versions
RUN pip install transformers==4.35.0 \
                numpy==1.24.3 \
                scikit-learn==1.3.0

# Copy code
COPY . /app
WORKDIR /app

# Run training
CMD ["python", "train.py", "--seed", "42", "--deterministic"]
```

---

## Citation for Reproducibility

If you successfully reproduce our results, please cite:

```bibtex
@article{riaz2024explainable,
  title={An eXplainable Approach to Abstractive Text Summarization Using External Knowledge},
  author={Riaz, Sumeer and Bashir, M. Bilal and Naqvi, Syed Ali Hassan},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024},
  note={Reproduced results available at [your-repo-url]}
}
```

---

## Contact for Reproduction Issues

Having trouble reproducing?

**Email:** sumeer33885@iqraisb.edu.pk

**Include:**
- Environment details
- Command used
- Observed results
- Expected results
- Error logs (if any)

We aim to respond within 48 hours.

---

## Acknowledgments

We thank the research community for emphasizing reproducibility.
All code, configs, and data splits are publicly available.
