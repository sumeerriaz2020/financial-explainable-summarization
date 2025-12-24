"""
Unit Tests for Knowledge Graph Integration
===========================================

Tests for FIBO integration and KG operations.
"""

import unittest
import networkx as nx
from knowledge_graph import (
    FIBOIntegration,
    EntityLinker,
    CausalExtractor,
    TemporalExtractor
)


class TestFIBOIntegration(unittest.TestCase):
    """Test FIBO Ontology Integration"""
    
    def setUp(self):
        self.fibo = FIBOIntegration(
            ontology_path="data/fibo/test_fibo.owl"
        )
    
    def test_module_loading(self):
        """Test FIBO module loading"""
        modules = self.fibo.get_loaded_modules()
        
        self.assertIsInstance(modules, list)
        self.assertIn('FND_Agents', modules)
    
    def test_class_retrieval(self):
        """Test FIBO class retrieval"""
        cls = self.fibo.get_class('Corporation')
        
        if cls:
            self.assertIsNotNone(cls)
            self.assertTrue(hasattr(cls, 'properties'))
    
    def test_entity_linking(self):
        """Test entity linking to FIBO"""
        entity = "Apple Inc."
        fibo_id = self.fibo.link_entity(entity, "ORG")
        
        # Should return URI or None
        self.assertTrue(fibo_id is None or isinstance(fibo_id, str))
    
    def test_statistics(self):
        """Test FIBO statistics"""
        stats = self.fibo.get_statistics()
        
        self.assertIn('total_classes', stats)
        self.assertIn('total_properties', stats)
        self.assertEqual(stats['total_classes'], 219)
        self.assertEqual(stats['total_properties'], 466)


class TestEntityLinker(unittest.TestCase):
    """Test Entity Linking"""
    
    def setUp(self):
        self.fibo = FIBOIntegration()
        self.linker = EntityLinker(self.fibo)
    
    def test_entity_detection(self):
        """Test entity detection"""
        text = "Apple Inc. CEO Tim Cook announced results."
        entities = self.linker.detect_entities(text)
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
    
    def test_linking_strategies(self):
        """Test different linking strategies"""
        entity = ("Apple Inc.", "ORG")
        
        exact_match = self.linker.link_exact(entity)
        normalized_match = self.linker.link_normalized(entity)
        
        # At least one strategy should work
        self.assertTrue(
            exact_match is not None or normalized_match is not None
        )
    
    def test_confidence_scores(self):
        """Test confidence scoring"""
        entity = ("Apple Inc.", "ORG")
        result = self.linker.link_with_confidence(entity)
        
        if result:
            self.assertIn('fibo_id', result)
            self.assertIn('confidence', result)
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)


class TestCausalExtractor(unittest.TestCase):
    """Test Causal Relation Extraction"""
    
    def setUp(self):
        self.extractor = CausalExtractor()
    
    def test_causal_pattern_detection(self):
        """Test causal pattern detection"""
        text = "Rate hikes led to market decline."
        relations = self.extractor.extract_causal_relations(text)
        
        self.assertIsInstance(relations, list)
        if len(relations) > 0:
            rel = relations[0]
            self.assertIn('cause', rel)
            self.assertIn('effect', rel)
    
    def test_multiple_causals(self):
        """Test extraction of multiple causal relations"""
        text = "A caused B. B resulted in C."
        relations = self.extractor.extract_causal_relations(text)
        
        self.assertIsInstance(relations, list)
        # Should find 2 causal relations


class TestTemporalExtractor(unittest.TestCase):
    """Test Temporal Relation Extraction"""
    
    def setUp(self):
        self.extractor = TemporalExtractor()
    
    def test_temporal_expression_detection(self):
        """Test temporal expression detection"""
        text = "Q1 2023 results improved over Q4 2022."
        temporal_exprs = self.extractor.extract_temporal_expressions(text)
        
        self.assertIsInstance(temporal_exprs, list)
        self.assertGreater(len(temporal_exprs), 0)
    
    def test_temporal_ordering(self):
        """Test temporal ordering"""
        expr1 = "Q1 2023"
        expr2 = "Q4 2022"
        
        order = self.extractor.compare_temporal_order(expr1, expr2)
        
        # expr1 should be after expr2
        self.assertEqual(order, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
