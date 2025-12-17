# Complete Directory Structure

```
financial-explainable-summarization/
│
├── README.md                      # Main project documentation
├── LICENSE                        # MIT License
├── setup.py                       # Package installation
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── .gitignore                     # Git ignore rules
│
├── algorithms/                    # 6 Core Algorithms (148 KB)
│   ├── __init__.py
│   ├── algorithm_1_kg_construction.py      # KG construction
│   ├── algorithm_2_hybrid_inference.py     # Hybrid inference
│   ├── algorithm_3_mesa.py                 # Multi-stakeholder
│   ├── algorithm_4_causal_explain.py       # Causal preservation
│   ├── algorithm_5_consensus.py            # Ensemble selection
│   ├── algorithm_6_training.py             # Multi-stage training
│   └── ALGORITHMS_README.md
│
├── models/                        # Model Architecture (76 KB)
│   ├── __init__.py
│   ├── dual_encoder.py            # Text + KG encoders
│   ├── knowledge_graph_encoder.py # GAT layers
│   ├── cross_modal_attention.py   # Cross-modal fusion
│   ├── hybrid_model.py            # Complete model
│   └── MODELS_README.md
│
├── explainability/                # 5 Explainability Frameworks (40 KB)
│   ├── __init__.py
│   ├── mesa.py                    # Multi-stakeholder
│   ├── causal_explain.py          # Causal chains
│   ├── adapt_eval.py              # Adaptive evaluation
│   ├── consensus.py               # Ensemble consensus
│   ├── temporal_explain.py        # Temporal consistency
│   └── EXPLAINABILITY_README.md
│
├── knowledge_graph/               # Knowledge Graph Resources (80 KB)
│   ├── __init__.py
│   ├── fibo_integration.py        # FIBO interface
│   ├── fibo_extensions.owl        # Custom extensions
│   ├── entity_linking.py          # Entity → FIBO linking
│   ├── causal_extraction.py       # Causal extraction
│   ├── temporal_annotation.py     # Temporal annotation
│   └── KNOWLEDGE_GRAPH_README.md
│
├── utils/                         # Utility Scripts (58 KB)
│   ├── __init__.py
│   ├── preprocessing.py           # Document preprocessing
│   ├── entity_recognition.py      # NER with FinBERT
│   ├── knowledge_graph_utils.py   # KG operations
│   ├── visualization.py           # Paper figures
│   └── UTILS_README.md
│
├── configs/                       # Configuration Files
│   ├── model_config.yaml          # Model hyperparameters
│   ├── training_config.yaml       # Training settings
│   └── fibo_config.yaml           # FIBO paths
│
├── examples/                      # Usage Examples
│   ├── basic_summarization.py     # Simple example
│   ├── with_knowledge_graph.py    # With KG integration
│   ├── multi_stakeholder.py       # Multi-stakeholder
│   └── complete_pipeline.py       # Full pipeline
│
├── tests/                         # Unit Tests
│   ├── __init__.py
│   ├── test_algorithms.py
│   ├── test_models.py
│   ├── test_explainability.py
│   ├── test_knowledge_graph.py
│   └── test_utils.py
│
├── docs/                          # Documentation
│   ├── installation.md
│   ├── usage.md
│   ├── api_reference.md
│   ├── paper_reproduction.md
│   └── troubleshooting.md
│
├── scripts/                       # Helper Scripts
│   ├── download_fibo.sh           # Download FIBO ontology
│   ├── prepare_data.py            # Data preparation
│   └── run_experiments.sh         # Run experiments
│
├── data/                          # Data Directory (gitignored)
│   ├── raw/                       # Raw documents
│   ├── processed/                 # Preprocessed data
│   └── fibo/                      # FIBO ontology files
│
├── checkpoints/                   # Model Checkpoints (gitignored)
│   ├── best_model.pt
│   └── stage_*.pt
│
├── results/                       # Results (gitignored)
│   ├── metrics.json
│   └── predictions.json
│
└── figures/                       # Generated Figures (gitignored)
    ├── table_ii_performance.png
    ├── table_iii_explainability.png
    └── error_analysis.png
```

## Directory Descriptions

### Source Code (402 KB total)

**`algorithms/`** - 6 core algorithms from the paper
- Knowledge graph construction with FIBO
- Hybrid neural-symbolic inference
- Multi-stakeholder explanation (MESA)
- Causal relationship extraction
- Interpretable consensus selection
- Multi-stage training pipeline

**`models/`** - Neural architecture components
- Dual encoder (text + KG)
- Graph attention networks (GAT)
- Cross-modal attention mechanisms
- Complete hybrid model

**`explainability/`** - 5 novel explainability frameworks
- MESA: Multi-stakeholder with RL
- CAUSAL-EXPLAIN: Causal preservation
- ADAPT-EVAL: Adaptive evaluation
- CONSENSUS: Ensemble selection
- TEMPORAL-EXPLAIN: Temporal consistency

**`knowledge_graph/`** - Knowledge graph resources
- FIBO 2024-Q1 integration
- Custom OWL extensions
- Entity linking utilities
- Causal and temporal extractors

**`utils/`** - Utility functions
- Document preprocessing
- Financial NER with FinBERT
- KG query and traversal
- Visualization generation

### Configuration & Setup

**`configs/`** - YAML configuration files
- Model hyperparameters
- Training settings
- FIBO paths and settings

**`examples/`** - Usage examples
- Basic summarization
- KG-integrated summarization
- Multi-stakeholder explanations
- Complete pipeline

### Testing & Documentation

**`tests/`** - Unit tests
- Algorithm tests
- Model tests
- Integration tests
- Coverage reports

**`docs/`** - Comprehensive documentation
- Installation guide
- Usage instructions
- API reference
- Paper reproduction guide

### Data & Outputs (gitignored)

**`data/`** - Dataset storage
- Raw financial documents
- Preprocessed data
- FIBO ontology files

**`checkpoints/`** - Model checkpoints
- Best model weights
- Stage-specific checkpoints
- Training state

**`results/`** - Experimental results
- Evaluation metrics
- Predictions
- Analysis outputs

**`figures/`** - Generated visualizations
- Performance charts
- Error analysis
- Attention heatmaps

## File Count Summary

```
Source Code:          402 KB (30+ files)
Configuration:        3 files
Examples:             4 files
Tests:                6+ files
Documentation:        5+ files
Setup Files:          5 files

Total Python Code:    ~6,500 lines
Total Documentation:  ~3,000 lines
```

## Quick Navigation

- **Getting Started:** See `README.md`
- **Installation:** See `requirements.txt` and `environment.yml`
- **Usage Examples:** See `examples/`
- **API Documentation:** See `docs/api_reference.md`
- **Algorithm Details:** See `algorithms/ALGORITHMS_README.md`
- **Model Architecture:** See `models/MODELS_README.md`
- **Explainability:** See `explainability/EXPLAINABILITY_README.md`

## Installation Steps

1. Clone repository
2. Create environment: `conda env create -f environment.yml`
3. Activate: `conda activate fin-explainable`
4. Install package: `pip install -e .`
5. Download FIBO: `bash scripts/download_fibo.sh`
6. Run tests: `pytest tests/`

## Ready to Use!

All components are production-ready and thoroughly documented.
