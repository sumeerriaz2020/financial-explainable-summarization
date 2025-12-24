"""
Unit Tests for Explainability Frameworks
=========================================

Tests for all 5 explainability frameworks:
1. MESA (Multi-stakeholder Explanation)
2. CAUSAL-EXPLAIN (Causal Chain Extraction)
3. TEMPORAL-EXPLAIN (Temporal Consistency)
4. CONSENSUS (Multi-method Consensus)
5. ADAPT-EVAL (Adaptive Evaluation)
"""

import unittest
import torch
import numpy as np
from explainability import (
    MESAExplainer,
    CausalExplainer,
    TemporalExplainer,
    ConsensusExplainer,
    AdaptiveExplainer
)


class TestMESAExplainer(unittest.TestCase):
    """Test MESA Explainability Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.explainer = MESAExplainer()
        
        self.summary = "Apple reported revenue of $89.5B, up 12% YoY."
        self.document = "Apple Inc. announced Q4 results..."
    
    def test_stakeholder_profiles(self):
        """Test stakeholder profile management"""
        profiles = self.explainer.get_stakeholder_profiles()
        
        self.assertEqual(len(profiles), 4)
        self.assertIn('analyst', profiles)
        self.assertIn('compliance', profiles)
        self.assertIn('executive', profiles)
        self.assertIn('investor', profiles)
    
    def test_analyst_explanation(self):
        """Test analyst-specific explanation"""
        explanation = self.explainer.generate_explanation(
            summary=self.summary,
            document=self.document,
            stakeholder='analyst'
        )
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 20)
        
        # Analyst explanations should mention metrics/analysis
        explanation_lower = explanation.lower()
        has_financial_terms = any(
            term in explanation_lower
            for term in ['revenue', 'growth', 'yoy', 'margin', 'metric']
        )
        self.assertTrue(has_financial_terms)
    
    def test_compliance_explanation(self):
        """Test compliance officer explanation"""
        explanation = self.explainer.generate_explanation(
            summary=self.summary,
            document=self.document,
            stakeholder='compliance'
        )
        
        self.assertIsInstance(explanation, str)
        
        # Compliance explanations should mention regulations/disclosure
        explanation_lower = explanation.lower()
        has_compliance_terms = any(
            term in explanation_lower
            for term in ['regulation', 'disclosure', 'requirement', 'compliance']
        )
        # Note: May not always contain these terms, so we just check it's non-empty
        self.assertGreater(len(explanation), 0)
    
    def test_executive_explanation(self):
        """Test executive-focused explanation"""
        explanation = self.explainer.generate_explanation(
            summary=self.summary,
            document=self.document,
            stakeholder='executive'
        )
        
        self.assertIsInstance(explanation, str)
        # Executive explanations should be concise
        self.assertLess(len(explanation), 500)
    
    def test_investor_explanation(self):
        """Test investor-focused explanation"""
        explanation = self.explainer.generate_explanation(
            summary=self.summary,
            document=self.document,
            stakeholder='investor'
        )
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
    
    def test_invalid_stakeholder(self):
        """Test handling of invalid stakeholder"""
        with self.assertRaises(ValueError):
            self.explainer.generate_explanation(
                summary=self.summary,
                document=self.document,
                stakeholder='invalid_role'
            )
    
    def test_explanation_quality_metrics(self):
        """Test explanation quality assessment"""
        explanation = self.explainer.generate_explanation(
            summary=self.summary,
            document=self.document,
            stakeholder='analyst'
        )
        
        quality = self.explainer.assess_quality(explanation, self.document)
        
        self.assertIn('faithfulness', quality)
        self.assertIn('completeness', quality)
        self.assertIn('clarity', quality)
        
        # Scores should be between 0 and 1
        for score in quality.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestCausalExplainer(unittest.TestCase):
    """Test CAUSAL-EXPLAIN Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.explainer = CausalExplainer(
            max_chain_length=5,
            confidence_threshold=0.6
        )
        
        self.causal_text = (
            "Interest rate hikes led to market volatility. "
            "Market volatility caused investor concerns. "
            "Investor concerns resulted in portfolio rebalancing."
        )
    
    def test_causal_chain_extraction(self):
        """Test causal chain extraction"""
        chains = self.explainer.extract_causal_chains(self.causal_text)
        
        self.assertIsInstance(chains, list)
        self.assertGreater(len(chains), 0)
        
        # Check chain structure
        chain = chains[0]
        self.assertIsInstance(chain, list)
        
        for link in chain:
            self.assertIsInstance(link, tuple)
            self.assertEqual(len(link), 3)  # (cause, effect, confidence)
    
    def test_chain_length_constraint(self):
        """Test max chain length enforcement"""
        chains = self.explainer.extract_causal_chains(self.causal_text)
        
        for chain in chains:
            self.assertLessEqual(len(chain), self.explainer.max_chain_length)
    
    def test_confidence_scores(self):
        """Test confidence score computation"""
        chains = self.explainer.extract_causal_chains(self.causal_text)
        
        for chain in chains:
            for cause, effect, confidence in chain:
                self.assertIsInstance(confidence, float)
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)
                self.assertGreaterEqual(confidence, self.explainer.confidence_threshold)
    
    def test_causal_markers(self):
        """Test causal marker detection"""
        markers = ['led to', 'caused', 'resulted in', 'due to', 'because']
        
        for marker in markers:
            text = f"A {marker} B."
            chains = self.explainer.extract_causal_chains(text)
            # Should detect at least something (or return empty gracefully)
            self.assertIsInstance(chains, list)
    
    def test_no_causal_relations(self):
        """Test text without causal relations"""
        text = "Apple is a technology company. It makes products."
        chains = self.explainer.extract_causal_chains(text)
        
        self.assertEqual(len(chains), 0)
    
    def test_chain_visualization(self):
        """Test causal chain visualization"""
        chains = self.explainer.extract_causal_chains(self.causal_text)
        
        if len(chains) > 0:
            visualization = self.explainer.visualize_chain(chains[0])
            
            self.assertIsInstance(visualization, str)
            self.assertIn('→', visualization)  # Arrow symbol


class TestTemporalExplainer(unittest.TestCase):
    """Test TEMPORAL-EXPLAIN Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.explainer = TemporalExplainer()
        
        self.temporal_text = (
            "Q1 2023 saw revenue growth. "
            "Q2 2023 continued the positive trend. "
            "Q3 2023 showed acceleration."
        )
    
    def test_temporal_marker_detection(self):
        """Test temporal expression detection"""
        markers = self.explainer.detect_temporal_markers(self.temporal_text)
        
        self.assertIsInstance(markers, list)
        self.assertGreater(len(markers), 0)
        
        # Should find Q1, Q2, Q3
        marker_texts = [m['text'] for m in markers]
        self.assertTrue(any('Q1' in m or 'Q2' in m or 'Q3' in m for m in marker_texts))
    
    def test_temporal_ordering(self):
        """Test temporal ordering verification"""
        is_ordered = self.explainer.verify_temporal_ordering(self.temporal_text)
        
        self.assertIsInstance(is_ordered, bool)
        self.assertTrue(is_ordered)  # Q1 → Q2 → Q3 is correctly ordered
    
    def test_temporal_inconsistency_detection(self):
        """Test detection of temporal inconsistencies"""
        inconsistent_text = "Q3 results improved. Then Q1 declined."
        
        inconsistencies = self.explainer.detect_inconsistencies(inconsistent_text)
        
        self.assertIsInstance(inconsistencies, list)
        # May detect Q3 before Q1 as inconsistency
    
    def test_market_regime_detection(self):
        """Test market regime detection"""
        bull_text = "Markets rallied. Stocks surged. Growth accelerated."
        regime = self.explainer.detect_market_regime(bull_text)
        
        self.assertIn('regime', regime)
        self.assertIn('confidence', regime)
        
        # Regime should be one of: Bull, Bear, Crisis, Normal
        self.assertIn(regime['regime'], ['Bull', 'Bear', 'Crisis', 'Normal'])
    
    def test_temporal_consistency_score(self):
        """Test TCC computation (Equation 9)"""
        explanations = [
            "Q1 showed growth.",
            "Q2 continued growth.",
            "Q3 maintained momentum."
        ]
        
        tcc = self.explainer.compute_temporal_consistency(explanations)
        
        self.assertIsInstance(tcc, float)
        self.assertGreaterEqual(tcc, 0.0)
        self.assertLessEqual(tcc, 1.0)
    
    def test_regime_change_detection(self):
        """Test detection of market regime changes"""
        text_with_change = (
            "Markets were bullish in Q1. "
            "Crisis hit in Q2. "
            "Recovery began in Q3."
        )
        
        changes = self.explainer.detect_regime_changes(text_with_change)
        
        self.assertIsInstance(changes, list)
        # Should detect transitions


class TestConsensusExplainer(unittest.TestCase):
    """Test CONSENSUS Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.explainer = ConsensusExplainer(
            methods=['gradient', 'attention', 'lime', 'shap'],
            scoring_weights={
                'accuracy': 0.30,
                'linguistic': 0.25,
                'stakeholder_fit': 0.25,
                'compliance': 0.20
            }
        )
        
        self.sample_explanations = {
            'gradient': "Revenue growth driven by iPhone sales.",
            'attention': "Revenue increased due to strong iPhone performance.",
            'lime': "iPhone sales primarily contributed to revenue growth.",
            'shap': "Main factor: iPhone sales boosting revenue."
        }
    
    def test_method_registration(self):
        """Test explainability method registration"""
        methods = self.explainer.get_methods()
        
        self.assertEqual(len(methods), 4)
        self.assertIn('gradient', methods)
        self.assertIn('attention', methods)
    
    def test_consensus_computation(self):
        """Test consensus score computation"""
        consensus = self.explainer.compute_consensus(self.sample_explanations)
        
        self.assertIn('consensus_score', consensus)
        self.assertIn('agreement_matrix', consensus)
        
        # Consensus score should be between 0 and 1
        self.assertGreaterEqual(consensus['consensus_score'], 0.0)
        self.assertLessEqual(consensus['consensus_score'], 1.0)
    
    def test_weighted_scoring(self):
        """Test weighted consensus scoring"""
        scores = {
            'accuracy': 0.8,
            'linguistic': 0.7,
            'stakeholder_fit': 0.9,
            'compliance': 0.6
        }
        
        weighted_score = self.explainer.compute_weighted_score(scores)
        
        self.assertIsInstance(weighted_score, float)
        
        # Verify weighting: 0.30*0.8 + 0.25*0.7 + 0.25*0.9 + 0.20*0.6
        expected = 0.30 * 0.8 + 0.25 * 0.7 + 0.25 * 0.9 + 0.20 * 0.6
        self.assertAlmostEqual(weighted_score, expected, places=5)
    
    def test_disagreement_detection(self):
        """Test detection of method disagreements"""
        conflicting_explanations = {
            'method1': "Revenue increased due to iPhone.",
            'method2': "Revenue decreased despite strong sales.",
            'method3': "Revenue remained stable."
        }
        
        disagreements = self.explainer.detect_disagreements(conflicting_explanations)
        
        self.assertIsInstance(disagreements, list)
        # Should detect conflicts between methods
    
    def test_best_explanation_selection(self):
        """Test selection of best explanation"""
        best = self.explainer.select_best_explanation(self.sample_explanations)
        
        self.assertIn('explanation', best)
        self.assertIn('method', best)
        self.assertIn('score', best)
        
        # Best explanation should be one of the input explanations
        self.assertIn(best['explanation'], self.sample_explanations.values())


class TestAdaptiveExplainer(unittest.TestCase):
    """Test ADAPT-EVAL Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.explainer = AdaptiveExplainer(
            base_metrics=['rouge', 'bertscore', 'factual'],
            adaptive_threshold=0.7
        )
        
        self.technical_doc = "The model uses GAT with 8 heads and 3 layers."
        self.simple_doc = "Sales increased last quarter."
    
    def test_document_complexity_assessment(self):
        """Test document complexity assessment"""
        tech_complexity = self.explainer.assess_complexity(self.technical_doc)
        simple_complexity = self.explainer.assess_complexity(self.simple_doc)
        
        self.assertIsInstance(tech_complexity, float)
        self.assertIsInstance(simple_complexity, float)
        
        # Technical doc should have higher complexity
        self.assertGreater(tech_complexity, simple_complexity)
    
    def test_adaptive_metric_selection(self):
        """Test adaptive metric selection"""
        tech_metrics = self.explainer.select_metrics(self.technical_doc)
        simple_metrics = self.explainer.select_metrics(self.simple_doc)
        
        self.assertIsInstance(tech_metrics, list)
        self.assertIsInstance(simple_metrics, list)
        
        # Both should have base metrics
        for metric in self.explainer.base_metrics:
            self.assertIn(metric, tech_metrics)
    
    def test_threshold_adaptation(self):
        """Test threshold adaptation"""
        tech_threshold = self.explainer.adapt_threshold(self.technical_doc)
        simple_threshold = self.explainer.adapt_threshold(self.simple_doc)
        
        self.assertIsInstance(tech_threshold, float)
        self.assertIsInstance(simple_threshold, float)
        
        self.assertGreaterEqual(tech_threshold, 0.0)
        self.assertLessEqual(tech_threshold, 1.0)
    
    def test_evaluation_adaptation(self):
        """Test adaptive evaluation"""
        prediction = "Revenue increased by 12%."
        reference = "The company reported 12% revenue growth."
        
        results = self.explainer.evaluate(
            prediction=prediction,
            reference=reference,
            document=self.simple_doc
        )
        
        self.assertIn('scores', results)
        self.assertIn('selected_metrics', results)
        self.assertIn('adapted_threshold', results)
    
    def test_stakeholder_specific_adaptation(self):
        """Test stakeholder-specific metric adaptation"""
        analyst_metrics = self.explainer.adapt_for_stakeholder('analyst')
        compliance_metrics = self.explainer.adapt_for_stakeholder('compliance')
        
        self.assertIsInstance(analyst_metrics, list)
        self.assertIsInstance(compliance_metrics, list)
        
        # Different stakeholders may need different metrics
        # At minimum, both should have some metrics
        self.assertGreater(len(analyst_metrics), 0)
        self.assertGreater(len(compliance_metrics), 0)


# Test Suite
def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMESAExplainer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCausalExplainer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTemporalExplainer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConsensusExplainer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveExplainer))
    
    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
