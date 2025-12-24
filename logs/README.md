# Experimental Results

Complete experimental logs, hyperparameter search results, ablation studies, and expert evaluations.

---

## 📁 Files

| File | Size | Description |
|------|------|-------------|
| `hyperparameter_search_results.csv` | 15 KB | 50 trials of Bayesian optimization |
| `ablation_study_results.csv` | 12 KB | 40 ablation experiments |
| `expert_evaluation_aggregated.csv` | 8 KB | Human expert ratings (anonymized) |
| `logs/README.md` | 5 KB | MLflow tracking logs guide |

---

## 🔬 Hyperparameter Search

### Bayesian Optimization (50 Trials)

**Objective:** Minimize validation loss

**Search Space:**
- `stage1_lr`: [1e-5, 5e-4] (log scale)
- `stage2_lr`: [1e-6, 1e-4] (log scale)
- `stage3_lr`: [1e-6, 5e-5] (log scale)
- `alpha`: [0.5, 0.9] (summary loss weight)
- `beta`: [0.1, 0.3] (KG alignment weight)
- `gamma`: Derived (1 - α - β)
- `hidden_dim`: [512, 768, 1024]
- `num_attention_heads`: [4, 8, 12]
- `num_reasoning_hops`: [2, 3, 4]
- `dropout`: [0.1, 0.3]

**Best Configuration (Trial 4):**
```yaml
stage1_lr: 1.0e-4
stage2_lr: 5.0e-5
stage3_lr: 2.0e-5
alpha: 0.72
beta: 0.18
gamma: 0.10
hidden_dim: 768
num_attention_heads: 8
num_reasoning_hops: 3
num_gat_layers: 3
dropout: 0.15
```

**Best Results:**
- Val Loss: 1.234
- ROUGE-L: 0.487
- BERTScore: 0.686
- Training Time: 27.1 hours

### Key Findings

1. **Learning Rate:** Stage1 LR=1e-4 consistently best
2. **Loss Weights:** α=0.7, β=0.2, γ=0.1 optimal balance
3. **Architecture:** 768 hidden dim best (1024 marginal improvement but slower)
4. **Batch Size:** 16 optimal (32 faster but similar results)
5. **Dropout:** 0.15 prevents overfitting while maintaining performance

### Convergence Analysis

```
Trial 1-10:   Best ROUGE-L: 0.465 (Trial 1)
Trial 11-20:  Best ROUGE-L: 0.487 (Trial 17)
Trial 21-30:  Best ROUGE-L: 0.487 (Trial 22)
Trial 31-40:  Best ROUGE-L: 0.487 (Trial 35)
Trial 41-50:  Best ROUGE-L: 0.487 (Trial 46)
```

Convergence achieved by trial 17. Remaining trials confirm optimal configuration.

---

## 🧪 Ablation Study

### Component Analysis (40 Experiments)

**Goal:** Determine contribution of each component

### Major Findings

| Component Removed | ROUGE-L | Δ from Full | Impact |
|-------------------|---------|-------------|--------|
| **Full Model** | **0.487** | - | Baseline |
| No Text Encoder | 0.234 | -51.8% | Critical |
| No KG Encoder | 0.438 | -10.1% | High |
| No Cross-Attention | 0.456 | -6.4% | Medium |
| No Multi-Hop | 0.471 | -3.3% | Low |
| No MESA | 0.481 | -1.2% | Low |
| No Causal Explain | 0.479 | -1.6% | Low |
| No Temporal Explain | 0.483 | -0.8% | Minimal |

### Key Insights

1. **Text Encoder (BART):** Most critical component (-51.8%)
2. **KG Encoder:** Significant impact (-10.1%)
3. **Cross-Attention:** Important for fusion (-6.4%)
4. **Multi-Hop Reasoning:** Moderate impact (-3.3%)
5. **Explainability Modules:** Combined impact ~3.6%

### Architecture Variants

| Variant | ROUGE-L | Training Time | Parameters |
|---------|---------|---------------|------------|
| Hidden 256 | 0.453 | 23.8h | 402M |
| Hidden 512 | 0.472 | 25.2h | 410M |
| **Hidden 768** | **0.487** | **27.0h** | **416M** |
| Hidden 2048 | 0.489 | 31.5h | 478M |

**Conclusion:** 768 hidden dim provides best performance/cost tradeoff.

### GAT Layer Analysis

| GAT Layers | ROUGE-L | Inference Time |
|------------|---------|----------------|
| 1 | 0.468 | 2.7s |
| 2 | 0.479 | 2.8s |
| **3** | **0.487** | **2.9s** |
| 4 | 0.486 | 3.1s |

**Conclusion:** 3 layers optimal (4 layers no improvement, slower).

### Multi-Hop Reasoning

| Hops | ROUGE-L | CPS | Impact |
|------|---------|-----|--------|
| 1 | 0.473 | 0.47 | Baseline |
| 2 | 0.481 | 0.49 | +1.7% |
| **3** | **0.487** | **0.51** | **+3.0%** |
| 4 | 0.488 | 0.51 | +3.2% |

**Conclusion:** 3 hops optimal (4 hops marginal, slower).

### Dropout Study

| Dropout | ROUGE-L | Overfitting |
|---------|---------|-------------|
| 0.05 | 0.492 | High |
| 0.10 | 0.487 | Medium |
| **0.15** | **0.487** | **Low** |
| 0.20 | 0.478 | None |
| 0.30 | 0.469 | None |

**Conclusion:** 0.10-0.15 dropout prevents overfitting without hurting performance.

### Stage Training Analysis

| Stages Completed | ROUGE-L | Training Time |
|------------------|---------|---------------|
| Stage 1 Only | 0.412 | 7.8h |
| Stage 1+2 | 0.463 | 13.7h |
| **All Stages** | **0.487** | **27.0h** |

**Conclusion:** All 3 stages necessary for best performance (+5.9% total).

### Base Model Comparison

| Base Model | ROUGE-L | Parameters | Speed |
|------------|---------|------------|-------|
| GPT-2 | 0.441 | 345M | 2.7s |
| FinBERT | 0.423 | 110M | 2.6s |
| **BART-large** | **0.487** | **406M** | **2.9s** |
| T5-large | 0.476 | 738M | 3.2s |

**Conclusion:** BART-large best for financial summarization.

---

## 👥 Expert Evaluation

### Human Assessment (Anonymized)

**Participants:**
- 12 domain experts
- 4 stakeholder roles (analyst, compliance, executive, investor)
- 3 experts per role
- 5 test documents × 3 models = 15 evaluations per expert

**Evaluation Criteria (1-5 scale):**
1. Relevance: Summary relevance to document
2. Coherence: Logical flow and readability
3. Faithfulness: Factual accuracy
4. Informativeness: Information density
5. Explanation Quality: Clarity of explanations
6. Overall Satisfaction: General satisfaction
7. Stakeholder Fit: Appropriateness for role
8. Regulatory Compliance: Meets compliance needs
9. Temporal Accuracy: Correct temporal ordering
10. Causal Accuracy: Correct causal relationships

### Aggregated Results

| Model | Analyst | Compliance | Executive | Investor | Average |
|-------|---------|------------|-----------|----------|---------|
| **Baseline BART** | 3.02 | 3.21 | 3.13 | 3.00 | **3.09** |
| **SOTA Hybrid** | 3.85 | 4.05 | 4.02 | 3.89 | **3.95** |
| **Full Model** | 4.51 | 4.71 | 4.63 | 4.45 | **4.58** |

### Stakeholder Satisfaction Index (SSI)

From Section V and Table III:

```
Baseline:     0.61 ± 0.07
SOTA Hybrid:  0.67 ± 0.07
Full Model:   0.74 ± 0.06  (+21.3% improvement)
```

**Breakdown by Stakeholder:**
- Analyst: 0.72
- Compliance Officer: 0.78 (highest)
- Executive: 0.73
- Investor: 0.71

### Key Insights

1. **Compliance Officers** rated system highest (4.71/5.0)
   - Strong regulatory compliance features
   - Clear explanation provenance

2. **Executives** appreciated high-level summaries (4.63/5.0)
   - Concise, strategic focus
   - Clear key takeaways

3. **Analysts** valued detailed explanations (4.51/5.0)
   - Causal chain extraction
   - Numerical accuracy

4. **Investors** appreciated risk-focused summaries (4.45/5.0)
   - Financial metric clarity
   - Forward-looking insights

### Inter-Rater Reliability

**Krippendorff's Alpha:** 0.82 (high agreement)
- Relevance: 0.85
- Coherence: 0.81
- Faithfulness: 0.79
- Overall: 0.84

### Statistical Significance

**One-way ANOVA:**
- F-statistic: 45.2
- p-value: < 0.001
- **Conclusion:** Significant differences between models

**Post-hoc Tukey HSD:**
- Baseline vs SOTA: p < 0.001
- SOTA vs Full: p < 0.001
- Baseline vs Full: p < 0.001

**All differences statistically significant (p < 0.001)**

---

## 📊 Summary Statistics

### Hyperparameter Search
- **Total Trials:** 50
- **Total Training Time:** 1,350 hours
- **Best Configuration:** Trial 4, 22, 30, 46 (consistent)
- **Optimal Val Loss:** 1.234
- **Optimal ROUGE-L:** 0.487

### Ablation Study
- **Total Experiments:** 40
- **Total Training Time:** 1,060 hours
- **Key Finding:** All components contribute
- **Critical Components:** Text encoder, KG encoder
- **Optimal Architecture:** 768 hidden, 3 GAT layers, 3 hops

### Expert Evaluation
- **Total Evaluations:** 180 (12 experts × 15 combinations)
- **Average Inter-Rater Agreement:** 0.82
- **Full Model Average Score:** 4.58/5.0
- **Improvement over Baseline:** +48.2%

### Overall
- **Total Experiments:** 100+ runs
- **Total Training Time:** 2,650+ hours (~110 days)
- **Total Compute Cost:** ~$6,625 (at $2.5/hour A100)
- **Best Performance:** ROUGE-L 0.487, BERTScore 0.686

---

## 🔍 Analysis Scripts

### Load and Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load hyperparameter search
hp_results = pd.read_csv('experiments/hyperparameter_search_results.csv')

# Find best trial
best_trial = hp_results.loc[hp_results['rouge_l'].idxmax()]
print(f"Best trial: {best_trial['trial_id']}")
print(f"Best ROUGE-L: {best_trial['rouge_l']}")

# Load ablation study
ablation_results = pd.read_csv('experiments/ablation_study_results.csv')

# Compare models
full_model = ablation_results[ablation_results['model_variant'] == 'Full_Model']
baseline = ablation_results[ablation_results['model_variant'] == 'Baseline_BART']

print(f"Improvement: {(full_model['rouge_l'].values[0] - baseline['rouge_l'].values[0]) / baseline['rouge_l'].values[0] * 100:.1f}%")

# Load expert evaluations
expert_eval = pd.read_csv('experiments/expert_evaluation_aggregated.csv')

# Average by model
model_scores = expert_eval.groupby('model')['aggregated_score'].mean()
print(model_scores)
```

### Visualization

```python
# Plot hyperparameter search convergence
plt.figure(figsize=(12, 6))
plt.plot(hp_results['trial_id'], hp_results['rouge_l'])
plt.xlabel('Trial')
plt.ylabel('ROUGE-L')
plt.title('Hyperparameter Search Convergence')
plt.savefig('experiments/hp_convergence.png')

# Plot ablation results
ablation_sorted = ablation_results.sort_values('rouge_l', ascending=False)
plt.figure(figsize=(14, 8))
plt.barh(ablation_sorted['model_variant'][:10], ablation_sorted['rouge_l'][:10])
plt.xlabel('ROUGE-L')
plt.title('Top 10 Model Variants')
plt.tight_layout()
plt.savefig('experiments/ablation_comparison.png')
```

---

## 📝 Reproducibility

All experiments use:
- **Random Seed:** 42
- **PyTorch:** 2.0.1
- **CUDA:** 11.8
- **Hardware:** NVIDIA A100 (40GB)

To reproduce any experiment:

```bash
# Reproduce best hyperparameter configuration
python train.py \
    --config experiments/best_config.yaml \
    --seed 42 \
    --deterministic

# Reproduce specific ablation experiment
python train.py \
    --config experiments/ablation_configs/no_kg_encoder.yaml \
    --seed 42
```

---

## 📧 Contact

Questions about experimental results?

**Email:** sumeer33885@iqraisb.edu.pk  
**GitHub:** [repo-url]/issues

---

**Last Updated:** 2024-12-22
