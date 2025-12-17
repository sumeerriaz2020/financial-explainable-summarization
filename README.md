# Financial Explainable Summarization with Hybrid Neural-Symbolic AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An eXplainable Approach to Abstractive Text Summarization Using External Knowledge: A Novel Framework for Financial Domain Applications**

Official implementation of the hybrid neural-symbolic framework for explainable financial document summarization with FIBO knowledge graph integration.

**Authors:** Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
**Affiliation:** IQRA University Islamabad Campus, Pakistan

---

## 🌟 Highlights

- **15.3% improvement** in ROUGE-L (0.421 → 0.497)
- **50% improvement** in Causal Preservation Score (0.34 → 0.51)
- **21.3% improvement** in Stakeholder Satisfaction (0.61 → 0.74)
- **FIBO 2024-Q1 integration** with 219 classes, 466 properties
- **5 novel explainability frameworks** (MESA, CAUSAL-EXPLAIN, ADAPT-EVAL, CONSENSUS, TEMPORAL-EXPLAIN)
- **Production-ready code** with comprehensive documentation

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Performance](#performance)
- [Usage Examples](#usage-examples)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This repository implements a **hybrid neural-symbolic framework** that combines:

1. **Dual-Encoder Architecture**: Separate processing of text (BART) and knowledge graph (GAT)
2. **FIBO Knowledge Graph**: Financial Industry Business Ontology with custom extensions
3. **Cross-Modal Attention**: Bidirectional text↔KG fusion (Equation 2)
4. **Multi-Hop Reasoning**: 3-hop traversal for complex relationships
5. **Multi-Stakeholder Explainability**: Role-specific explanations (analyst, compliance, executive)

### Novel Contributions

✅ **MESA Framework**: Multi-stakeholder explanation with RL-based preference learning  
✅ **CAUSAL-EXPLAIN**: Causal chain extraction and preservation (CPS metric)  
✅ **ADAPT-EVAL**: Adaptive evaluation with continuous user feedback  
✅ **INTERPRETABLE-CONSENSUS**: Transparent ensemble explanation selection  
✅ **TEMPORAL-EXPLAIN**: Market regime-aware temporal consistency (TCC metric)

---

## 🏗️ Architecture

```
Input Document
      ↓
  [Text Encoder (BART)]  ←→  [KG Encoder (GAT)]
      ↓                           ↓
      └──→ [Cross-Modal Attention] ←──┘
                  ↓
         [Multi-Hop Reasoning]
                  ↓
              [Decoder]
                  ↓
             Summary + Explanation
```

**Components:**
- **Algorithms**: 6 core algorithms (KG construction, hybrid inference, MESA, etc.)
- **Models**: Dual encoder, KG encoder, cross-modal attention, complete hybrid model
- **Explainability**: 5 novel frameworks for stakeholder-aware explanations
- **Knowledge Graph**: FIBO integration + custom extensions (temporal, causal, regime)
- **Utils**: Preprocessing, NER, KG operations, visualization

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU)

### Option 1: pip install (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/financial-explainable-summarization.git
cd financial-explainable-summarization

# Install package
pip install -e .
```

### Option 2: Conda environment

```bash
# Create environment
conda env create -f environment.yml
conda activate fin-explainable

# Install package
pip install -e .
```

### Option 3: Manual installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "import fin_explainable; print(fin_explainable.__version__)"
```

---

## ⚡ Quick Start

### 1. Basic Summarization

```python
from fin_explainable.models import HybridModel
from fin_explainable.knowledge_graph import FIBOIntegration
from transformers import AutoTokenizer

# Initialize
model = HybridModel(
    text_encoder_name="facebook/bart-large",
    kg_input_dim=768,
    num_reasoning_hops=3
)

fibo = FIBOIntegration()
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

# Prepare input
text = "Apple Inc. reported Q4 earnings of $89.5B..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=128,
    num_beams=4
)

decoded = tokenizer.decode(summary[0], skip_special_tokens=True)
print(decoded)
```

### 2. With Knowledge Graph Integration

```python
from fin_explainable.algorithms import KnowledgeGraphConstructor
from fin_explainable.knowledge_graph import EntityLinker, CausalExtractor

# Build KG
constructor = KnowledgeGraphConstructor(fibo_path="path/to/fibo.owl")
kg = constructor.construct_knowledge_graph(documents)

# Extract entities and link to FIBO
linker = EntityLinker(fibo)
entities = linker.link_entities_batch(entity_mentions, text)

# Extract causal relationships
causal = CausalExtractor()
causal_chains = causal.extract_causal_relations(text)

# Generate with KG
summary = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    kg_node_features=kg_features,
    kg_adjacency=kg_adj,
    kg_paths=reasoning_paths
)
```

### 3. Multi-Stakeholder Explanations

```python
from fin_explainable.explainability import MESAExplainer

# Initialize MESA
mesa = MESAExplainer()

# Generate stakeholder-specific explanation
profile = mesa.profiles['analyst']
explanation = mesa.generate_explanation(
    summary=summary_text,
    document=document,
    kg=knowledge_graph,
    stakeholder=profile
)

print(explanation)

# Update with feedback
mesa.update_with_feedback(profile, explanation, feedback_score=0.85)
```

### 4. Complete Pipeline

```python
from fin_explainable import CompletePipeline

# Initialize pipeline
pipeline = CompletePipeline(
    model_name="facebook/bart-large",
    fibo_path="path/to/fibo.owl",
    device='cuda'
)

# Process document
result = pipeline.process(
    text=document_text,
    stakeholder='analyst',
    generate_explanation=True
)

print(f"Summary: {result['summary']}")
print(f"Explanation: {result['explanation']}")
print(f"Causal Chains: {len(result['causal_chains'])}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## 📁 Repository Structure

```
financial-explainable-summarization/
├── algorithms/                    # 6 core algorithms (148 KB)
│   ├── algorithm_1_kg_construction.py
│   ├── algorithm_2_hybrid_inference.py
│   ├── algorithm_3_mesa.py
│   ├── algorithm_4_causal_explain.py
│   ├── algorithm_5_consensus.py
│   ├── algorithm_6_training.py
│   └── ALGORITHMS_README.md
│
├── models/                        # Model architecture (76 KB)
│   ├── dual_encoder.py
│   ├── knowledge_graph_encoder.py
│   ├── cross_modal_attention.py
│   ├── hybrid_model.py
│   └── MODELS_README.md
│
├── explainability/                # 5 explainability frameworks (40 KB)
│   ├── mesa.py
│   ├── causal_explain.py
│   ├── adapt_eval.py
│   ├── consensus.py
│   ├── temporal_explain.py
│   └── EXPLAINABILITY_README.md
│
├── knowledge_graph/               # KG resources (80 KB)
│   ├── fibo_integration.py
│   ├── fibo_extensions.owl
│   ├── entity_linking.py
│   ├── causal_extraction.py
│   ├── temporal_annotation.py
│   └── KNOWLEDGE_GRAPH_README.md
│
├── utils/                         # Utilities (58 KB)
│   ├── preprocessing.py
│   ├── entity_recognition.py
│   ├── knowledge_graph_utils.py
│   ├── visualization.py
│   └── UTILS_README.md
│
├── configs/                       # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── fibo_config.yaml
│
├── examples/                      # Usage examples
│   ├── basic_summarization.py
│   ├── with_knowledge_graph.py
│   ├── multi_stakeholder.py
│   └── complete_pipeline.py
│
├── tests/                         # Unit tests
│   └── test_*.py
│
├── docs/                          # Documentation
│   ├── installation.md
│   ├── usage.md
│   └── api_reference.md
│
├── setup.py                       # Package setup
├── requirements.txt               # Dependencies
├── environment.yml                # Conda environment
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

**Total:** 402 KB of production-ready code

---

## ✨ Key Features

### Algorithms
- ✅ **Algorithm 1**: Knowledge Graph Construction (FIBO integration)
- ✅ **Algorithm 2**: Hybrid Neural-Symbolic Inference
- ✅ **Algorithm 3**: MESA Framework (Multi-Stakeholder)
- ✅ **Algorithm 4**: CAUSAL-EXPLAIN (Causal Preservation)
- ✅ **Algorithm 5**: INTERPRETABLE-CONSENSUS (Ensemble)
- ✅ **Algorithm 6**: Multi-Stage Training Pipeline

### Models
- ✅ Dual Encoder (Text + KG)
- ✅ Graph Attention Networks (GAT)
- ✅ Cross-Modal Attention (Bidirectional)
- ✅ Multi-Hop Reasoning (3 hops)
- ✅ Complete Hybrid Architecture

### Explainability
- ✅ MESA: Stakeholder-aware explanations with RL
- ✅ CAUSAL-EXPLAIN: Causal chain extraction (CPS metric)
- ✅ ADAPT-EVAL: Adaptive evaluation with feedback
- ✅ CONSENSUS: Transparent ensemble selection
- ✅ TEMPORAL-EXPLAIN: Market regime-aware consistency

### Knowledge Graph
- ✅ FIBO 2024-Q1 integration (219 classes, 466 properties)
- ✅ Custom extensions (temporal, causal, stakeholder, regime)
- ✅ Entity linking to FIBO concepts
- ✅ Causal relationship extraction
- ✅ Temporal annotation and regime detection

---

## 📊 Performance

### Summarization Quality (Table II)

| Metric | Baseline | SOTA | **Our System** | Improvement |
|--------|----------|------|----------------|-------------|
| ROUGE-L | 0.421 | 0.465 | **0.497** | **+15.3%** |
| BERTScore | 0.612 | 0.658 | **0.686** | **+12.1%** |
| Factual Consistency | 83.7% | 86.2% | **87.6%** | **+4.7pp** |

### Explainability Metrics (Table III)

| Metric | Baseline | SOTA | **Our System** | Improvement |
|--------|----------|------|----------------|-------------|
| SSI | 0.61±0.07 | 0.67±0.07 | **0.74±0.06** | **+21.3%** |
| CPS | 0.34±0.08 | 0.42±0.08 | **0.51±0.07** | **+50.0%** |
| TCC | 0.38±0.06 | 0.45±0.07 | **0.54±0.07** | **+42.1%** |
| Consistency | 0.43±0.07 | 0.51±0.07 | **0.59±0.06** | **+37.2%** |

### Computational Costs

| Metric | Baseline | Our System | Overhead |
|--------|----------|------------|----------|
| Inference Time | 2.4s | 2.9s | +21% |
| Memory Usage | 8.2 GB | 9.8 GB | +19.5% |
| Training Time | 24h | 36h | +50% |

---

## 💡 Usage Examples

### Training

```bash
# Train complete model
python train.py \
    --config configs/training_config.yaml \
    --data_path data/financial_corpus \
    --output_dir checkpoints/

# Multi-stage training (Algorithm 6)
python train.py \
    --multi_stage \
    --stage1_epochs 3 \
    --stage2_epochs 2 \
    --stage3_epochs 5
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/test.json \
    --metrics rouge bertscore factual

# Generate visualizations
python utils/visualization.py \
    --results results.json \
    --output_dir figures/
```

### Inference

```bash
# Single document
python inference.py \
    --model checkpoints/best_model.pt \
    --input document.txt \
    --stakeholder analyst

# Batch processing
python inference.py \
    --model checkpoints/best_model.pt \
    --input_dir documents/ \
    --output_dir summaries/ \
    --batch_size 8
```

---

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [Algorithm Details](algorithms/ALGORITHMS_README.md)
- [Model Architecture](models/MODELS_README.md)
- [Explainability Frameworks](explainability/EXPLAINABILITY_README.md)

---

## 🐛 Known Limitations

1. **Error Rate**: 56.5% total error rate (21.8% high-severity)
2. **Scalability**: Documents >50 pages show degradation
3. **KG Size**: Performance degrades with >5000 nodes
4. **Computational Cost**: 21% slower inference, 19.5% higher memory
5. **Domain**: Optimized for financial documents (requires adaptation for other domains)

See Section V.D in the paper for detailed error analysis.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone with development dependencies
git clone https://github.com/yourusername/financial-explainable-summarization.git
cd financial-explainable-summarization

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{riaz2024explainable,
  title={An eXplainable Approach to Abstractive Text Summarization Using External Knowledge: A Novel Framework for Financial Domain Applications},
  author={Riaz, Sumeer and Bashir, M. Bilal and Naqvi, Syed Ali Hassan},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contact

**Sumeer Riaz**  
Email: sumeer33885@iqraisb.edu.pk  
Affiliation: IQRA University Islamabad Campus

**Dr. M. Bilal Bashir**  
Email: bilal.bashir@iqraisb.edu.pk

**Syed Ali Hassan Naqvi**  
Email: ali.hassan@iqraisb.edu.pk

---

## 🙏 Acknowledgments

- FIBO (Financial Industry Business Ontology) by EDM Council
- Hugging Face Transformers
- PyTorch Team
- NetworkX Community

---

## 📈 Project Status

- [x] Core algorithms implementation
- [x] Model architecture
- [x] Explainability frameworks
- [x] Knowledge graph integration
- [x] Utility scripts
- [x] Documentation
- [ ] Pre-trained model weights (coming soon)
- [ ] Online demo (coming soon)
- [ ] Extended domain support (future work)

---

**⭐ If you find this work useful, please star the repository!**

---

<p align="center">
  Made with ❤️ by the IQRA University Research Team
</p>
