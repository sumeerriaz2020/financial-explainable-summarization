# System Limitations

Clear articulation of system limitations and known issues.

---

## Error Rates (Section V.D from paper)

### Overall Error Analysis

**Total Error Rate:** **56.5%**
- High-severity errors: **21.8%**
- Medium-severity errors: **22.3%**
- Low-severity errors: **12.4%**

### Error Distribution by Category

| Category | Percentage | Severity | Description |
|----------|------------|----------|-------------|
| **Entity Misidentification** | 21.8% | High | Incorrect entity extraction or linking to FIBO |
| **Causal Misattribution** | 15.2% | High | Incorrect causal relationships identified |
| **Temporal Inconsistency** | 12.5% | Medium | Timeline or temporal ordering errors |
| **Factual Errors** | 7.0% | High | Incorrect facts or hallucinations |
| **Other** | 43.5% | Variable | Miscellaneous errors |

---

## Performance Limitations

### Computational Costs

**Training:**
- Time: 26-30 hours (single A100) | 8 hours (4x A100)
- Cost: $75-225
- Memory: 24 GB GPU, 64 GB RAM
- Storage: 100 GB

**Inference:**
- Time: 2.9s per document (+21% vs baseline)
- Memory: 9.8 GB (+19.5% vs baseline)
- Parameters: 416M

**Overhead vs BART baseline:**
- Inference time: +21%
- Memory usage: +19.5%
- Training time: +50%

### Scalability Issues

1. **Document Length:** Performance degrades for documents >50 pages
2. **Knowledge Graph Size:** Optimal performance with <5,000 nodes; degrades with >5,000 nodes
3. **Batch Size:** Limited by GPU memory (max 16 on A100 40GB)

---

## Domain Limitations

### Financial Domain Specificity

**Optimized for:**
- ✅ Quarterly earnings reports
- ✅ 10-K/10-Q SEC filings
- ✅ Financial news articles
- ✅ Analyst reports

**NOT optimized for:**
- ❌ General news summarization
- ❌ Scientific papers
- ❌ Legal documents
- ❌ Non-financial business documents

**Adaptation Required:**
- Fine-tuning on domain-specific data
- Custom FIBO module selection
- Stakeholder profile customization

### Language Support

- **Supported:** English only
- **Limitation:** Financial terminology in English
- **Future Work:** Multi-lingual support requires:
  - Multi-lingual BART model
  - Translated FIBO ontology
  - Language-specific entity recognition

---

## Knowledge Graph Limitations

### FIBO Coverage

**Covered Entities:**
- Financial instruments: 87% coverage
- Organizations: 82% coverage
- Relationships: 78% coverage

**Not Covered:**
- Emerging financial products
- Non-standard entities
- Domain-specific jargon
- Informal terminology

### Entity Linking Issues

**Accuracy:** 87% overall
- Exact match: 95%
- Normalized match: 82%
- Type-based: 75%
- Keyword-based: 68%

**Common Errors:**
- Ambiguous entity names
- New/unlisted companies
- Non-standard abbreviations
- Informal references

---

## Explainability Limitations

### Stakeholder Profiles

**Predefined Only:**
- Analyst
- Compliance Officer
- Executive
- Investor

**Limitations:**
- Cannot dynamically create new profiles
- Fixed information needs per profile
- Requires manual update for new stakeholder types

### Causal Explanation

**Causal Preservation Score (CPS):** 0.51
- **Issue:** 49% of causal chains not fully preserved
- **Cause:** Complex causal reasoning beyond model capacity
- **Impact:** May miss subtle cause-effect relationships

### Temporal Consistency

**Temporal Consistency Coefficient (TCC):** 0.54
- **Issue:** 46% inconsistency in temporal explanations
- **Cause:** Market regime detection accuracy limits
- **Impact:** May produce contradictory temporal statements

---

## Data Requirements

### Training Data

**Minimum Requirements:**
- Documents: 1,000+ financial documents
- Summaries: Human-written reference summaries
- Quality: Professional-grade annotations
- Diversity: Multiple companies, sectors, time periods

**Limitations:**
- Requires domain expertise for annotation
- Expensive to acquire quality data
- Privacy concerns with proprietary documents
- Data distribution shift affects performance

### Knowledge Graph Requirements

- FIBO ontology (2GB download)
- Custom extensions (50 MB)
- Entity linking database
- Preprocessing time: ~2 hours for 1,000 documents

---

## Model Limitations

### Architecture Constraints

1. **Fixed Context Window:** 1024 tokens maximum
   - **Impact:** Long documents truncated
   - **Workaround:** Document chunking required

2. **KG Node Limit:** 5,000 nodes optimal
   - **Impact:** Large graphs cause slowdown
   - **Workaround:** Graph pruning needed

3. **Multi-hop Reasoning:** 3 hops maximum
   - **Impact:** Can't capture distant relationships
   - **Workaround:** Pre-compute important paths

### Generation Quality

**Known Issues:**
- Factual hallucinations: 12.4% of summaries
- Repetition: Occasional duplicate phrases
- Generic language: Sometimes lacks specificity
- Number accuracy: ±5% error in numerical values

---

## Regulatory Compliance Limitations

### GDPR Compliance

- ✅ Explanation provenance tracked
- ✅ User data not retained
- ⚠️ Model may memorize training data
- ❌ Cannot guarantee complete data deletion

### EU AI Act Compliance

- ✅ Risk assessment conducted
- ✅ Human oversight enabled
- ✅ Transparency documentation
- ⚠️ High-risk use requires additional validation

---

## Known Bugs & Issues

### Critical Issues

1. **Memory Leak:** Long-running inference may accumulate memory
   - **Workaround:** Restart server periodically
   - **Status:** Under investigation

2. **KG Construction Slowdown:** >10,000 documents
   - **Workaround:** Batch processing with restarts
   - **Status:** Optimization planned

### Minor Issues

1. Special characters in entity names cause linking errors
2. Very short summaries (<20 words) have lower quality
3. Numerical tables not well-handled
4. PDF extraction may miss formatting

---

## Not Supported

### Features NOT Implemented

- ❌ Real-time streaming summarization
- ❌ Interactive refinement
- ❌ Multi-document summarization
- ❌ Cross-lingual summarization
- ❌ Automatic fact-checking
- ❌ Source attribution per sentence
- ❌ Custom evaluation metrics
- ❌ Online learning / incremental training

### Future Work Required

1. **Reduce error rate** from 56.5% to <30%
2. **Improve causal preservation** from 51% to >80%
3. **Expand FIBO coverage** to emerging financial instruments
4. **Multi-lingual support**
5. **Real-time inference** (<1s per document)
6. **Automatic stakeholder profile learning**

---

## Risk Assessment

### High-Risk Scenarios

**DO NOT USE for:**
- Fully automated investment decisions
- Legal compliance determinations
- Regulatory filings without human review
- Medical/healthcare financial decisions

**Reason:** 56.5% error rate and 12.4% factual errors require human verification

### Recommended Use Cases

**SAFE for:**
- Draft summary generation (with human review)
- Information retrieval assistance
- Research and analysis support
- Educational purposes

**With Safeguards:**
- Human-in-the-loop validation
- Confidence thresholding
- Multiple model consensus
- Expert domain review

---

## Disclaimer

This system is a research prototype. It should **NOT** be used as the sole basis for:
- Financial decisions
- Regulatory compliance
- Legal determinations
- High-stakes business decisions

**Always verify outputs with domain experts.**

---

## Reporting Issues

Found a limitation not listed here?

**Report to:**
- GitHub Issues: [repo-url]/issues
- Email: sumeer33885@iqraisb.edu.pk
- Include: Error description, input data (if shareable), system logs

---

## Version History

- **v1.0.0 (2024-12):** Initial release
  - Total error rate: 56.5%
  - CPS: 0.51
  - TCC: 0.54

Future versions will track limitation improvements.
