# Configuration Files

Complete configuration system for the hybrid neural-symbolic model with FIBO integration.

---

## 📁 Files (42 KB total)

| File | Size | Description |
|------|------|-------------|
| `model_config.yaml` | 5.6 KB | Model architecture & hyperparameters |
| `training_config.yaml` | 7.1 KB | Multi-stage training settings |
| `fibo_modules.yaml` | 8.8 KB | FIBO ontology configuration |
| `environment.yml` | 1.6 KB | Conda environment (from repo root) |
| `requirements.txt` | 0.3 KB | Python dependencies (from repo root) |

---

## 🎯 Configuration Overview

### 1. Model Configuration (`model_config.yaml`)

**Purpose:** Complete model architecture specification

**Key Sections:**

#### **Model Architecture**
```yaml
model:
  name: "HybridFinancialSummarizer"
  version: "1.0.0"
  
  text_encoder:
    model_name: "facebook/bart-large"
    hidden_dim: 1024
    num_layers: 12
  
  kg_encoder:
    hidden_dim: 512
    num_layers: 3
    num_heads: 8
  
  cross_attention:
    num_heads: 8
    bidirectional: true
    use_gating: true
  
  multi_hop_reasoning:
    num_hops: 3
    hidden_dim: 512
```

#### **Model Size**
```yaml
model_size:
  total_parameters: 416000000     # 416M total
  bart_parameters: 406000000       # 406M (97.6%)
  kg_encoder_parameters: 2100000   # 2.1M (0.5%)
  cross_attention_parameters: 4700000  # 4.7M (1.1%)
  multi_hop_parameters: 3200000    # 3.2M (0.8%)
```

#### **Compute Requirements**
```yaml
compute:
  train_batch_size: 16
  train_memory_gb: 24
  inference_batch_size: 1
  inference_memory_gb: 9.8
  inference_time_per_doc: 2.9  # seconds
```

#### **Explainability Components**
```yaml
explainability:
  mesa:
    enabled: true
    stakeholder_profiles: ["analyst", "compliance", "executive", "investor"]
  
  causal_explain:
    enabled: true
    max_chain_length: 5
  
  temporal_explain:
    enabled: true
    detect_regime: true
```

**Usage:**
```python
import yaml

with open('configs/model_config.yaml') as f:
    config = yaml.safe_load(f)

model = HybridModel(**config['model'])
```

---

### 2. Training Configuration (`training_config.yaml`)

**Purpose:** Multi-stage training pipeline (Algorithm 6)

**Three Training Stages:**

#### **Stage 1: Warm-up (3 epochs)**
```yaml
stage1:
  name: "warm-up"
  epochs: 3
  learning_rate: 1.0e-4
  trainable_components: ["decoder"]
  frozen_components: ["text_encoder", "kg_encoder"]
  
  loss_weights:
    summary: 1.0
    kg_alignment: 0.0
    explanation: 0.0
```

#### **Stage 2: KG Integration (2 epochs)**
```yaml
stage2:
  name: "kg-integration"
  epochs: 2
  learning_rate: 5.0e-5
  trainable_components: ["text_encoder", "kg_encoder", "cross_attention"]
  
  loss_weights:
    summary: 0.7   # α
    kg_alignment: 0.3  # β
    explanation: 0.0
```

#### **Stage 3: End-to-End (5 epochs)**
```yaml
stage3:
  name: "end-to-end"
  epochs: 5
  learning_rate: 2.0e-5
  trainable_components: ["all"]
  
  loss_weights:
    summary: 0.7      # α
    kg_alignment: 0.2  # β
    explanation: 0.1   # γ
```

**Loss Function (Equation 10):**
```
L_total = α*L_summary + β*L_KG + γ*L_explanation
```

**Data Configuration:**
```yaml
data:
  train_path: "data/train.json"
  val_path: "data/val.json"
  test_path: "data/test.json"
  max_source_length: 1024
  max_target_length: 128
  kg_path: "data/knowledge_graphs/"
```

**Optimization:**
```yaml
optimization:
  optimizer: "AdamW"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  
  scheduler:
    type: "linear_with_warmup"
    warmup_ratio: 0.1
  
  max_grad_norm: 1.0
```

**Expected Training Time:**
```yaml
expected_duration:
  stage1: "8 hours"
  stage2: "6 hours"
  stage3: "12 hours"
  total: "26-30 hours"  # on A100 40GB
```

**Expected Performance (Table II):**
```yaml
expected_results:
  rouge_l: 0.487
  bertscore: 0.686
  factual_consistency: 87.6
  ssi: 0.74
  cps: 0.51
  tcc: 0.54
```

**Usage:**
```python
import yaml

with open('configs/training_config.yaml') as f:
    config = yaml.safe_load(f)

trainer = MultiStageTrainer(
    model=model,
    config=config['training']
)
```

---

### 3. FIBO Modules Configuration (`fibo_modules.yaml`)

**Purpose:** FIBO ontology module selection and settings

**FIBO Version:**
```yaml
fibo:
  version: "2024-Q1"
  release_date: "2024-01-15"
  base_uri: "https://spec.edmcouncil.org/fibo/ontology"
```

**Core Modules (Table I from paper):**

#### **Module 1: FND/Agents**
```yaml
FND_Agents:
  enabled: true
  statistics:
    classes: 73
    properties: 152
  
  key_classes:
    - "Organization"
    - "Person"
    - "ExecutiveOfficer"
    - "Shareholder"
  
  usage_priority: "high"
```

#### **Module 2: BE/LegalEntities**
```yaml
BE_LegalEntities:
  enabled: true
  statistics:
    classes: 47
    properties: 84
  
  key_classes:
    - "Corporation"
    - "PubliclyTradedCompany"
    - "Subsidiary"
  
  usage_priority: "high"
```

#### **Module 3: FND/Relations**
```yaml
FND_Relations:
  enabled: true
  statistics:
    classes: 31
    properties: 103
  
  key_classes:
    - "Control"
    - "Ownership"
    - "Causality"
  
  usage_priority: "high"
```

#### **Module 4: FBC/Products**
```yaml
FBC_Products:
  enabled: true
  statistics:
    classes: 68
    properties: 127
  
  key_classes:
    - "FinancialInstrument"
    - "Security"
    - "Equity"
    - "Derivative"
  
  usage_priority: "high"
```

**Total Statistics:**
```yaml
statistics:
  total_classes: 219
  total_properties: 466
  entity_coverage: 0.87  # 87% coverage
```

**Custom Extensions:**
```yaml
extensions:
  enabled: true
  
  temporal:
    enabled: true
    classes: ["TemporalRelationship"]
  
  causal:
    enabled: true
    classes: ["CausalRelationship"]
  
  stakeholder:
    enabled: true
    individuals: ["FinancialAnalyst", "ComplianceOfficer"]
  
  regulatory:
    enabled: true
    individuals: ["GDPR_Article13", "EU_AI_Act_Article52"]
```

**Loading Configuration:**
```yaml
loading:
  strategy: "lazy"
  cache_enabled: true
  cache_size_mb: 500
  parallel: true
  num_workers: 4
```

**Usage:**
```python
import yaml

with open('configs/fibo_modules.yaml') as f:
    config = yaml.safe_load(f)

fibo = FIBOIntegration(config['fibo'])
```

---

## 🚀 Quick Start

### 1. Load All Configurations

```python
import yaml
from pathlib import Path

def load_config(config_name):
    """Load a configuration file"""
    config_path = Path('configs') / f'{config_name}.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

# Load all configs
model_config = load_config('model_config')
training_config = load_config('training_config')
fibo_config = load_config('fibo_modules')
```

### 2. Initialize Components

```python
from models import HybridModel
from knowledge_graph import FIBOIntegration
from training import MultiStageTrainer

# Initialize FIBO
fibo = FIBOIntegration(fibo_config['fibo'])

# Initialize model
model = HybridModel(
    **model_config['model'],
    fibo_integration=fibo
)

# Initialize trainer
trainer = MultiStageTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=training_config['training']
)
```

### 3. Start Training

```python
# Train with multi-stage pipeline
trainer.train(output_dir='checkpoints/')
```

---

## 📊 Configuration Matrix

### Model Configurations

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Text Encoder** | BART-large | 1024 hidden dim, 12 layers |
| **KG Encoder** | GAT | 512 hidden dim, 3 layers, 8 heads |
| **Cross-Attention** | Bidirectional | 8 heads, gated fusion |
| **Multi-Hop** | 3 hops | Attention-based aggregation |
| **Total Parameters** | 416M | BART: 406M + Custom: 10M |

### Training Configurations

| Stage | Epochs | LR | Components | Loss Weights (α, β, γ) |
|-------|--------|-----|-----------|------------------------|
| **Stage 1** | 3 | 1e-4 | Decoder only | (1.0, 0.0, 0.0) |
| **Stage 2** | 2 | 5e-5 | Encoder+KG | (0.7, 0.3, 0.0) |
| **Stage 3** | 5 | 2e-5 | All | (0.7, 0.2, 0.1) |

### FIBO Module Statistics

| Module | Classes | Properties | Priority |
|--------|---------|------------|----------|
| **FND/Agents** | 73 | 152 | High |
| **BE/LegalEntities** | 47 | 84 | High |
| **FND/Relations** | 31 | 103 | High |
| **FBC/Products** | 68 | 127 | High |
| **TOTAL** | **219** | **466** | - |

---

## 🔧 Customization Guide

### Adjust Batch Size

**For 24GB GPU:**
```yaml
# training_config.yaml
batching:
  train_batch_size: 16
  gradient_accumulation_steps: 1
```

**For 16GB GPU:**
```yaml
batching:
  train_batch_size: 8
  gradient_accumulation_steps: 2
```

**For 12GB GPU:**
```yaml
batching:
  train_batch_size: 4
  gradient_accumulation_steps: 4
```

### Adjust Model Size

**Reduce memory (faster, less accurate):**
```yaml
# model_config.yaml
model:
  kg_encoder:
    hidden_dim: 256  # instead of 512
    num_layers: 2    # instead of 3
  
  multi_hop_reasoning:
    num_hops: 2      # instead of 3
```

### Enable/Disable FIBO Modules

```yaml
# fibo_modules.yaml
modules:
  FND_Agents:
    enabled: true    # Core module
  
  FBC_Markets:
    enabled: false   # Optional module
```

### Adjust Training Duration

**Quick experiment (2 hours):**
```yaml
# training_config.yaml
training:
  stage1: {epochs: 1}
  stage2: {epochs: 1}
  stage3: {epochs: 1}
```

**Full training (30 hours):**
```yaml
training:
  stage1: {epochs: 3}
  stage2: {epochs: 2}
  stage3: {epochs: 5}
```

---

## 📝 Configuration Validation

### Validate Configurations

```python
def validate_config(config, schema):
    """Validate configuration against schema"""
    # Check required fields
    required_fields = schema.get('required', [])
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Check data types
    # ... additional validation
    
    return True

# Validate model config
validate_config(model_config, model_schema)
```

### Check Compatibility

```python
def check_compatibility(model_config, training_config):
    """Check if configs are compatible"""
    
    # Check hidden dimensions match
    text_dim = model_config['model']['text_encoder']['hidden_dim']
    kg_output = model_config['model']['kg_encoder']['output_dim']
    
    if text_dim != kg_output:
        raise ValueError(
            f"Dimension mismatch: text_encoder={text_dim}, "
            f"kg_encoder={kg_output}"
        )
    
    # Check batch sizes
    # ... additional checks
    
    return True
```

---

## 🎯 Best Practices

### 1. **Start with Default Configs**
- Use provided configs as baseline
- Test with small dataset first
- Gradually customize

### 2. **Version Control Configs**
```bash
git add configs/*.yaml
git commit -m "Add configuration files"
```

### 3. **Document Changes**
```yaml
# model_config.yaml
# Modified: 2024-12-22
# Changes: Increased KG encoder layers from 2 to 3
# Reason: Improve knowledge graph encoding
```

### 4. **Use Config Inheritance**
```python
# base_config.yaml (base settings)
# experiment_1.yaml (overrides)

import yaml

def load_with_inheritance(config_path, base_path):
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f)
    
    # Merge configs
    return {**base, **override}
```

---

## ✅ Testing Configurations

```bash
# Test model config
python -c "
from models import HybridModel
import yaml
config = yaml.safe_load(open('configs/model_config.yaml'))
model = HybridModel(**config['model'])
print('Model config valid!')
"

# Test training config
python -c "
import yaml
config = yaml.safe_load(open('configs/training_config.yaml'))
print('Training config valid!')
"

# Test FIBO config
python -c "
from knowledge_graph import FIBOIntegration
import yaml
config = yaml.safe_load(open('configs/fibo_modules.yaml'))
fibo = FIBOIntegration(config['fibo'])
print('FIBO config valid!')
"
```

---

## 🚀 Ready to Configure!

All configuration files are production-ready with comprehensive documentation!

**Next Steps:**
1. Review and customize configs for your setup
2. Validate configurations
3. Start training with: `python train.py --config configs/training_config.yaml`
