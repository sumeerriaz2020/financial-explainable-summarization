"""
Evaluation Metrics
==================

Comprehensive metrics for evaluating summarization quality and explainability:
- ROUGE (L, 1, 2)
- BERTScore
- Factual Consistency
- SSI (Stakeholder Satisfaction Index)
- CPS (Causal Preservation Score)
- TCC (Temporal Consistency Coefficient)
- RCS (Regulat Compliance Score)

Reference: Section V (Evaluation Results), Tables II, III
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROUGEMetrics:
    """ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)"""
    
    def __init__(self):
        """Initialize ROUGE scorer"""
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.available = True
        except ImportError:
            logger.warning("rouge_score not available")
            self.available = False
    
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            Dictionary of ROUGE scores
        """
        if not self.available:
            return self._compute_manual(predictions, references)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def _compute_manual(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Manual ROUGE-L implementation"""
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # LCS
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            
            # Precision and recall
            if len(pred_tokens) > 0:
                precision = lcs_length / len(pred_tokens)
            else:
                precision = 0.0
            
            if len(ref_tokens) > 0:
                recall = lcs_length / len(ref_tokens)
            else:
                recall = 0.0
            
            # F-measure
            if precision + recall > 0:
                f_measure = 2 * precision * recall / (precision + recall)
            else:
                f_measure = 0.0
            
            scores.append(f_measure)
        
        return {'rougeL': np.mean(scores)}
    
    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """Longest Common Subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class BERTScoreMetrics:
    """BERTScore for semantic similarity"""
    
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        """Initialize BERTScore"""
        try:
            from bert_score import BERTScorer
            self.scorer = BERTScorer(
                model_type=model_type,
                lang="en",
                rescale_with_baseline=True
            )
            self.available = True
        except ImportError:
            logger.warning("bert_score not available")
            self.available = False
    
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            BERTScore (precision, recall, F1)
        """
        if not self.available:
            logger.warning("BERTScore not available, returning dummy scores")
            return {'bertscore': 0.0}
        
        P, R, F1 = self.scorer.score(predictions, references)
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item(),
            'bertscore': F1.mean().item()  # Main metric
        }


class FactualConsistencyMetrics:
    """Factual consistency evaluation"""
    
    def compute(
        self,
        predictions: List[str],
        sources: List[str],
        extracted_facts: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Compute factual consistency score
        
        Args:
            predictions: Generated summaries
            sources: Source documents
            extracted_facts: Pre-extracted facts (optional)
            
        Returns:
            Factual consistency score
        """
        scores = []
        
        for pred, source in zip(predictions, sources):
            # Extract facts from prediction
            pred_facts = self._extract_facts(pred)
            
            # Extract facts from source
            source_facts = self._extract_facts(source)
            
            # Compute overlap
            if len(pred_facts) > 0:
                consistent = sum(
                    1 for fact in pred_facts
                    if self._is_supported(fact, source_facts)
                )
                consistency = consistent / len(pred_facts)
            else:
                consistency = 1.0
            
            scores.append(consistency)
        
        return {
            'factual_consistency': np.mean(scores) * 100  # Percentage
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract facts (simplified: sentences with numbers/entities)"""
        import re
        
        sentences = text.split('.')
        facts = []
        
        for sent in sentences:
            sent = sent.strip()
            # Check if contains numbers or capitalized entities
            if re.search(r'\d+|\$|%', sent) or re.search(r'\b[A-Z][a-z]+\b', sent):
                facts.append(sent.lower())
        
        return facts
    
    def _is_supported(self, fact: str, source_facts: List[str]) -> bool:
        """Check if fact is supported by source"""
        fact_words = set(fact.split())
        
        for source_fact in source_facts:
            source_words = set(source_fact.split())
            # Check overlap
            overlap = len(fact_words & source_words) / len(fact_words) if fact_words else 0
            if overlap > 0.5:
                return True
        
        return False


class SSIMetrics:
    """
    Stakeholder Satisfaction Index (SSI)
    
    Equation 5: SSI = Σ w_i * s_i
    where w_i is stakeholder weight, s_i is satisfaction score
    
    Reference: Algorithm 3 (MESA), Table III
    """
    
    def __init__(self):
        """Initialize SSI metrics"""
        # Stakeholder weights (from paper)
        self.stakeholder_weights = {
            'analyst': 0.30,
            'compliance': 0.25,
            'executive': 0.25,
            'investor': 0.20
        }
    
    def compute(
        self,
        explanations: Dict[str, List[str]],
        stakeholder_ratings: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Compute SSI score
        
        Args:
            explanations: Stakeholder-specific explanations
            stakeholder_ratings: Human ratings per stakeholder
            
        Returns:
            SSI scores
        """
        ssi_scores = []
        
        for i in range(len(next(iter(explanations.values())))):
            weighted_sum = 0.0
            
            for stakeholder, weight in self.stakeholder_weights.items():
                if stakeholder in stakeholder_ratings:
                    rating = stakeholder_ratings[stakeholder][i]
                    weighted_sum += weight * rating
            
            ssi_scores.append(weighted_sum)
        
        return {
            'ssi': np.mean(ssi_scores),
            'ssi_std': np.std(ssi_scores),
            'ssi_by_stakeholder': {
                stakeholder: np.mean(ratings)
                for stakeholder, ratings in stakeholder_ratings.items()
            }
        }


class CPSMetrics:
    """
    Causal Preservation Score (CPS)
    
    Equations 7-8: Standard and weighted CPS
    
    Reference: Algorithm 4 (CAUSAL-EXPLAIN), Table III
    """
    
    def compute(
        self,
        predicted_chains: List[List[Tuple]],
        reference_chains: List[List[Tuple]],
        importance_weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute CPS
        
        Args:
            predicted_chains: Predicted causal chains
            reference_chains: Reference causal chains
            importance_weights: Optional importance weights
            
        Returns:
            CPS scores
        """
        standard_cps = []
        weighted_cps = []
        
        for pred, ref in zip(predicted_chains, reference_chains):
            # Standard CPS (Equation 7)
            if len(ref) > 0:
                preserved = sum(
                    1 for chain in pred
                    if self._chain_matches(chain, ref)
                )
                std_score = preserved / len(ref)
            else:
                std_score = 1.0
            
            standard_cps.append(std_score)
            
            # Weighted CPS (Equation 8)
            if importance_weights:
                weighted_score = self._compute_weighted_cps(
                    pred, ref, importance_weights
                )
                weighted_cps.append(weighted_score)
        
        result = {
            'cps': np.mean(standard_cps),
            'cps_std': np.std(standard_cps)
        }
        
        if importance_weights:
            result['cps_weighted'] = np.mean(weighted_cps)
        
        return result
    
    def _chain_matches(
        self,
        pred_chain: List[Tuple],
        ref_chains: List[List[Tuple]]
    ) -> bool:
        """Check if predicted chain matches any reference chain"""
        for ref_chain in ref_chains:
            if self._chains_similar(pred_chain, ref_chain):
                return True
        return False
    
    def _chains_similar(
        self,
        chain1: List[Tuple],
        chain2: List[Tuple],
        threshold: float = 0.5
    ) -> bool:
        """Check if two chains are similar"""
        if len(chain1) != len(chain2):
            return False
        
        matches = sum(
            1 for c1, c2 in zip(chain1, chain2)
            if c1[0] == c2[0] and c1[1] == c2[1]  # cause and effect match
        )
        
        return matches / len(chain1) >= threshold
    
    def _compute_weighted_cps(
        self,
        pred: List,
        ref: List,
        weights: List[float]
    ) -> float:
        """Compute weighted CPS (Equation 8)"""
        if len(ref) == 0:
            return 1.0
        
        total_weight = sum(weights[:len(ref)])
        weighted_sum = 0.0
        
        for i, ref_chain in enumerate(ref):
            if self._chain_matches([ref_chain], pred):
                weighted_sum += weights[i]
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class TCCMetrics:
    """
    Temporal Consistency Coefficient (TCC)
    
    Equation 9: TCC = (1/N) Σ cos_sim(E_t, E_{t-1})
    
    Reference: TEMPORAL-EXPLAIN framework, Table III
    """
    
    def compute(
        self,
        explanation_sequence: List[str],
        embeddings: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute TCC
        
        Args:
            explanation_sequence: Temporal sequence of explanations
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            TCC score
        """
        if embeddings is None:
            embeddings = self._compute_embeddings(explanation_sequence)
        
        if len(embeddings) < 2:
            return {'tcc': 1.0}
        
        # Compute cosine similarities
        similarities = []
        
        for i in range(1, len(embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i-1].unsqueeze(0),
                embeddings[i].unsqueeze(0)
            )
            similarities.append(sim.item())
        
        return {
            'tcc': np.mean(similarities),
            'tcc_std': np.std(similarities),
            'tcc_min': np.min(similarities),
            'tcc_max': np.max(similarities)
        }
    
    def _compute_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Compute embeddings for texts (simplified)"""
        # In production, use actual sentence embeddings
        embeddings = []
        
        for text in texts:
            # Simple word-based embedding
            words = text.lower().split()
            # Random embedding for demonstration
            emb = torch.randn(768)  # BERT-base dimension
            embeddings.append(emb)
        
        return embeddings


class RCSMetrics:
    """Regulatory Compliance Score"""
    
    def compute(
        self,
        explanations: List[str],
        requirements: List[str]
    ) -> Dict[str, float]:
        """
        Compute regulatory compliance score
        
        Args:
            explanations: Generated explanations
            requirements: Regulatory requirements
            
        Returns:
            RCS score
        """
        scores = []
        
        for explanation in explanations:
            # Check requirements
            satisfied = sum(
                1 for req in requirements
                if self._satisfies_requirement(explanation, req)
            )
            
            score = satisfied / len(requirements) if requirements else 1.0
            scores.append(score)
        
        return {
            'rcs': np.mean(scores) * 100,  # Percentage
            'rcs_std': np.std(scores) * 100
        }
    
    def _satisfies_requirement(self, explanation: str, requirement: str) -> bool:
        """Check if explanation satisfies requirement"""
        # Simplified check: keyword matching
        keywords = requirement.lower().split()
        explanation_lower = explanation.lower()
        
        matches = sum(1 for kw in keywords if kw in explanation_lower)
        return matches / len(keywords) > 0.5


# Combined evaluator
class ComprehensiveEvaluator:
    """All metrics in one place"""
    
    def __init__(self):
        """Initialize all metrics"""
        self.rouge = ROUGEMetrics()
        self.bertscore = BERTScoreMetrics()
        self.factual = FactualConsistencyMetrics()
        self.ssi = SSIMetrics()
        self.cps = CPSMetrics()
        self.tcc = TCCMetrics()
        self.rcs = RCSMetrics()
        
        logger.info("Comprehensive Evaluator initialized")
    
    def evaluate_all(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute all metrics"""
        results = {}
        
        # Summarization metrics
        results.update(self.rouge.compute(predictions, references))
        results.update(self.bertscore.compute(predictions, references))
        results.update(self.factual.compute(predictions, sources))
        
        # Explainability metrics (if data provided)
        if 'explanations' in kwargs and 'stakeholder_ratings' in kwargs:
            results.update(self.ssi.compute(
                kwargs['explanations'],
                kwargs['stakeholder_ratings']
            ))
        
        if 'predicted_chains' in kwargs and 'reference_chains' in kwargs:
            results.update(self.cps.compute(
                kwargs['predicted_chains'],
                kwargs['reference_chains']
            ))
        
        if 'explanation_sequence' in kwargs:
            results.update(self.tcc.compute(kwargs['explanation_sequence']))
        
        return results


# Example usage
if __name__ == "__main__":
    print("Evaluation Metrics")
    print("=" * 70)
    
    print("\nAvailable Metrics:")
    print("  Summarization Quality:")
    print("    - ROUGE-1, ROUGE-2, ROUGE-L")
    print("    - BERTScore")
    print("    - Factual Consistency")
    
    print("\n  Explainability (Novel):")
    print("    - SSI: Stakeholder Satisfaction Index (Eq. 5)")
    print("    - CPS: Causal Preservation Score (Eq. 7-8)")
    print("    - TCC: Temporal Consistency Coefficient (Eq. 9)")
    print("    - RCS: Regulatory Compliance Score")
    
    print("\n" + "=" * 70)
    print("Metrics ready!")
    print("=" * 70)
