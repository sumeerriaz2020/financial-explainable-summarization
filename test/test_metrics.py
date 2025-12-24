"""
Unit Tests for Evaluation Metrics
==================================

Tests for all evaluation metrics including novel ones.
"""

import unittest
import numpy as np
from evaluation.metrics import (
    ROUGEMetrics,
    BERTScoreMetrics,
    FactualConsistencyMetrics,
    SSIMetrics,
    CPSMetrics,
    TCCMetrics,
    RCSMetrics
)


class TestROUGEMetrics(unittest.TestCase):
    """Test ROUGE Metrics"""
    
    def setUp(self):
        self.rouge = ROUGEMetrics()
        
        self.predictions = [
            "Apple reported strong earnings growth.",
            "Revenue increased by 12 percent."
        ]
        
        self.references = [
            "Apple Inc. announced robust earnings growth.",
            "The company reported 12% revenue increase."
        ]
    
    def test_rouge_computation(self):
        """Test ROUGE score computation"""
        scores = self.rouge.compute(self.predictions, self.references)
        
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        
        # Scores should be between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_perfect_match(self):
        """Test ROUGE with perfect match"""
        pred = ["This is a test sentence."]
        ref = ["This is a test sentence."]
        
        scores = self.rouge.compute(pred, ref)
        
        # Perfect match should give 1.0
        self.assertAlmostEqual(scores['rougeL'], 1.0, places=2)


class TestBERTScoreMetrics(unittest.TestCase):
    """Test BERTScore Metrics"""
    
    def setUp(self):
        self.bertscore = BERTScoreMetrics()
        
        self.predictions = ["Apple reported earnings growth."]
        self.references = ["Apple announced earnings increase."]
    
    def test_bertscore_computation(self):
        """Test BERTScore computation"""
        scores = self.bertscore.compute(self.predictions, self.references)
        
        self.assertIn('bertscore', scores)
        
        # BERTScore should be between 0 and 1
        self.assertGreaterEqual(scores['bertscore'], 0.0)
        self.assertLessEqual(scores['bertscore'], 1.0)
    
    def test_semantic_similarity(self):
        """Test semantic similarity detection"""
        # Semantically similar but different words
        pred = ["The company did well."]
        ref = ["The organization performed strongly."]
        
        scores = self.bertscore.compute(pred, ref)
        
        # Should have decent score due to semantic similarity
        self.assertGreater(scores['bertscore'], 0.5)


class TestFactualConsistencyMetrics(unittest.TestCase):
    """Test Factual Consistency Metrics"""
    
    def setUp(self):
        self.factual = FactualConsistencyMetrics()
        
        self.predictions = ["Apple reported $89.5B revenue."]
        self.sources = ["Apple Inc. announced revenue of $89.5 billion."]
    
    def test_factual_consistency(self):
        """Test factual consistency computation"""
        scores = self.factual.compute(self.predictions, self.sources)
        
        self.assertIn('factual_consistency', scores)
        self.assertGreaterEqual(scores['factual_consistency'], 0.0)
        self.assertLessEqual(scores['factual_consistency'], 100.0)
    
    def test_hallucination_detection(self):
        """Test detection of hallucinated facts"""
        pred = ["Apple reported $999B revenue."]  # Hallucinated
        src = ["Apple Inc. announced revenue of $89.5 billion."]
        
        scores = self.factual.compute(pred, src)
        
        # Should have lower consistency
        self.assertLess(scores['factual_consistency'], 100.0)


class TestSSIMetrics(unittest.TestCase):
    """Test Stakeholder Satisfaction Index (Equation 5)"""
    
    def setUp(self):
        self.ssi = SSIMetrics()
        
        self.explanations = {
            'analyst': ["Detailed analysis..."],
            'compliance': ["Meets requirements..."],
            'executive': ["High-level summary..."],
            'investor': ["Risk assessment..."]
        }
        
        self.stakeholder_ratings = {
            'analyst': [0.85],
            'compliance': [0.90],
            'executive': [0.80],
            'investor': [0.75]
        }
    
    def test_ssi_computation(self):
        """Test SSI computation (Equation 5)"""
        scores = self.ssi.compute(self.explanations, self.stakeholder_ratings)
        
        self.assertIn('ssi', scores)
        self.assertIn('ssi_std', scores)
        
        # SSI should be between 0 and 1
        self.assertGreaterEqual(scores['ssi'], 0.0)
        self.assertLessEqual(scores['ssi'], 1.0)
    
    def test_stakeholder_weights(self):
        """Test stakeholder weight application"""
        weights = self.ssi.stakeholder_weights
        
        # Weights should sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_weighted_average(self):
        """Test weighted average computation"""
        # Manual computation
        expected = (
            0.30 * 0.85 +  # analyst
            0.25 * 0.90 +  # compliance
            0.25 * 0.80 +  # executive
            0.20 * 0.75    # investor
        )
        
        scores = self.ssi.compute(self.explanations, self.stakeholder_ratings)
        
        self.assertAlmostEqual(scores['ssi'], expected, places=2)


class TestCPSMetrics(unittest.TestCase):
    """Test Causal Preservation Score (Equations 7-8)"""
    
    def setUp(self):
        self.cps = CPSMetrics()
        
        self.predicted_chains = [
            [("A", "B", 0.9), ("B", "C", 0.8)],
            [("X", "Y", 0.7)]
        ]
        
        self.reference_chains = [
            [("A", "B", 1.0), ("B", "C", 1.0)],
            [("X", "Y", 1.0)]
        ]
    
    def test_cps_computation(self):
        """Test CPS computation (Equation 7)"""
        scores = self.cps.compute(
            self.predicted_chains,
            self.reference_chains
        )
        
        self.assertIn('cps', scores)
        self.assertIn('cps_std', scores)
        
        # CPS should be between 0 and 1
        self.assertGreaterEqual(scores['cps'], 0.0)
        self.assertLessEqual(scores['cps'], 1.0)
    
    def test_weighted_cps(self):
        """Test weighted CPS (Equation 8)"""
        importance_weights = [0.9, 0.7]
        
        scores = self.cps.compute(
            self.predicted_chains,
            self.reference_chains,
            importance_weights=importance_weights
        )
        
        self.assertIn('cps_weighted', scores)
    
    def test_perfect_preservation(self):
        """Test perfect causal preservation"""
        chains = [[("A", "B", 1.0)]]
        
        scores = self.cps.compute(chains, chains)
        
        # Perfect match should give 1.0
        self.assertAlmostEqual(scores['cps'], 1.0, places=2)


class TestTCCMetrics(unittest.TestCase):
    """Test Temporal Consistency Coefficient (Equation 9)"""
    
    def setUp(self):
        self.tcc = TCCMetrics()
        
        self.explanation_sequence = [
            "Q1 showed strong growth.",
            "Q2 continued the momentum.",
            "Q3 maintained performance."
        ]
    
    def test_tcc_computation(self):
        """Test TCC computation (Equation 9)"""
        scores = self.tcc.compute(self.explanation_sequence)
        
        self.assertIn('tcc', scores)
        self.assertIn('tcc_std', scores)
        
        # TCC should be between 0 and 1
        self.assertGreaterEqual(scores['tcc'], 0.0)
        self.assertLessEqual(scores['tcc'], 1.0)
    
    def test_temporal_consistency(self):
        """Test temporal consistency measurement"""
        # Consistent sequence
        consistent = [
            "Markets improved in Q1.",
            "Growth continued in Q2.",
            "Momentum sustained in Q3."
        ]
        
        scores_consistent = self.tcc.compute(consistent)
        
        # Inconsistent sequence
        inconsistent = [
            "Markets crashed.",
            "Markets rallied.",
            "Markets crashed."
        ]
        
        scores_inconsistent = self.tcc.compute(inconsistent)
        
        # Consistent should have higher TCC
        self.assertGreater(
            scores_consistent['tcc'],
            scores_inconsistent['tcc']
        )


class TestRCSMetrics(unittest.TestCase):
    """Test Regulatory Compliance Score"""
    
    def setUp(self):
        self.rcs = RCSMetrics()
        
        self.explanations = [
            "Disclosure meets GDPR Article 13 requirements.",
            "Explanation satisfies EU AI Act Article 52."
        ]
        
        self.requirements = [
            "GDPR Article 13",
            "EU AI Act Article 52"
        ]
    
    def test_rcs_computation(self):
        """Test RCS computation"""
        scores = self.rcs.compute(self.explanations, self.requirements)
        
        self.assertIn('rcs', scores)
        self.assertIn('rcs_std', scores)
        
        # RCS should be percentage (0-100)
        self.assertGreaterEqual(scores['rcs'], 0.0)
        self.assertLessEqual(scores['rcs'], 100.0)
    
    def test_requirement_satisfaction(self):
        """Test requirement satisfaction checking"""
        explanation = "Meets GDPR requirements."
        requirement = "GDPR Article 13"
        
        satisfies = self.rcs._satisfies_requirement(explanation, requirement)
        
        self.assertIsInstance(satisfies, bool)
    
    def test_full_compliance(self):
        """Test full compliance scenario"""
        # Explanations that mention all requirements
        full_compliance_expls = [
            req for req in self.requirements
        ]
        
        scores = self.rcs.compute(full_compliance_expls, self.requirements)
        
        # Should have high RCS
        self.assertGreater(scores['rcs'], 50.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
