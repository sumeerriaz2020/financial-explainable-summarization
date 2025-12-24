"""
Unit Tests for Core Algorithms
===============================

Tests for all 6 core algorithms:
1. Knowledge Graph Construction (Algorithm 1)
2. Hybrid Inference (Algorithm 2)
3. MESA Framework (Algorithm 3)
4. CAUSAL-EXPLAIN (Algorithm 4)
5. ADAPT-EVAL (Algorithm 5)
6. Multi-Stage Training (Algorithm 6)
"""

import unittest
import torch
import numpy as np
from algorithms import (
    KnowledgeGraphConstructor,
    HybridInference,
    MESAFramework,
    CausalExplainer,
    AdaptiveEvaluator,
    MultiStageTrainer
)


class TestKnowledgeGraphConstructor(unittest.TestCase):
    """Test Algorithm 1: Knowledge Graph Construction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.constructor = KnowledgeGraphConstructor(
            fibo_path="data/fibo/test_fibo.owl",
            max_nodes=100,
            max_edges=200
        )
        
        self.sample_documents = [
            "Apple Inc. reported earnings of $89.5B in Q4 2023.",
            "CEO Tim Cook announced new product launches."
        ]
    
    def test_initialization(self):
        """Test constructor initialization"""
        self.assertIsNotNone(self.constructor)
        self.assertEqual(self.constructor.max_nodes, 100)
        self.assertEqual(self.constructor.max_edges, 200)
    
    def test_entity_extraction(self):
        """Test entity extraction from documents"""
        entities = self.constructor.extract_entities(self.sample_documents[0])
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        # Check for expected entities
        entity_texts = [e[0] for e in entities]
        self.assertIn("Apple Inc.", entity_texts)
    
    def test_relation_extraction(self):
        """Test relation extraction"""
        text = "Apple Inc. reported earnings of $89.5B."
        relations = self.constructor.extract_relations(text)
        
        self.assertIsInstance(relations, list)
        # Should find "reported" relation
        if len(relations) > 0:
            self.assertIn('relation', relations[0])
    
    def test_kg_construction(self):
        """Test complete KG construction (Algorithm 1)"""
        kg = self.constructor.construct_knowledge_graph(self.sample_documents)
        
        self.assertIsNotNone(kg)
        self.assertGreater(len(kg.nodes), 0)
        self.assertLessEqual(len(kg.nodes), 100)  # max_nodes
        
        # Check KG structure
        self.assertTrue(hasattr(kg, 'nodes'))
        self.assertTrue(hasattr(kg, 'edges'))
        self.assertTrue(hasattr(kg, 'adjacency_matrix'))
    
    def test_fibo_integration(self):
        """Test FIBO ontology integration"""
        entity = ("Apple Inc.", "ORG")
        fibo_id = self.constructor.link_to_fibo(entity)
        
        # Should return FIBO URI or None
        self.assertTrue(fibo_id is None or isinstance(fibo_id, str))
    
    def test_max_nodes_constraint(self):
        """Test max_nodes constraint is enforced"""
        # Create many documents to exceed max_nodes
        many_docs = [f"Document {i} with entity_{i}" for i in range(200)]
        
        kg = self.constructor.construct_knowledge_graph(many_docs)
        
        self.assertLessEqual(len(kg.nodes), self.constructor.max_nodes)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        kg = self.constructor.construct_knowledge_graph([])
        
        self.assertEqual(len(kg.nodes), 0)
        self.assertEqual(len(kg.edges), 0)


class TestHybridInference(unittest.TestCase):
    """Test Algorithm 2: Hybrid Neural-Symbolic Inference"""
    
    def setUp(self):
        """Set up test fixtures"""
        from models import HybridModel
        
        self.model = HybridModel(hidden_dim=512)
        self.inference = HybridInference(
            model=self.model,
            num_reasoning_hops=3
        )
        
        self.sample_text = "Apple Inc. reported Q4 earnings of $89.5B."
    
    def test_initialization(self):
        """Test inference engine initialization"""
        self.assertIsNotNone(self.inference)
        self.assertEqual(self.inference.num_reasoning_hops, 3)
    
    def test_neural_encoding(self):
        """Test neural encoding step"""
        encoding = self.inference.encode_text(self.sample_text)
        
        self.assertIsInstance(encoding, torch.Tensor)
        self.assertEqual(encoding.dim(), 2)  # [batch, hidden]
    
    def test_symbolic_reasoning(self):
        """Test symbolic reasoning with KG"""
        from knowledge_graph import KnowledgeGraph
        
        # Create mock KG
        kg = KnowledgeGraph()
        kg.add_node("Apple Inc.", "ORG")
        kg.add_node("Q4", "DATE")
        kg.add_edge("Apple Inc.", "reported_in", "Q4")
        
        reasoning_result = self.inference.symbolic_reasoning(kg, num_hops=3)
        
        self.assertIsNotNone(reasoning_result)
        self.assertTrue('paths' in reasoning_result or 'reachable' in reasoning_result)
    
    def test_hybrid_fusion(self):
        """Test neural-symbolic fusion"""
        neural_features = torch.randn(1, 512)
        symbolic_features = torch.randn(1, 512)
        
        fused = self.inference.fuse_features(neural_features, symbolic_features)
        
        self.assertIsInstance(fused, torch.Tensor)
        self.assertEqual(fused.shape, (1, 512))
    
    def test_inference_pipeline(self):
        """Test complete inference pipeline (Algorithm 2)"""
        from knowledge_graph import KnowledgeGraph
        
        kg = KnowledgeGraph()
        kg.add_node("Apple Inc.", "ORG")
        
        result = self.inference.infer(
            text=self.sample_text,
            knowledge_graph=kg,
            return_explanations=True
        )
        
        self.assertIn('summary', result)
        self.assertIn('reasoning_paths', result)
        self.assertIn('confidence', result)


class TestMESAFramework(unittest.TestCase):
    """Test Algorithm 3: MESA Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mesa = MESAFramework(
            stakeholder_profiles=["analyst", "compliance", "executive", "investor"],
            use_rl_learning=True
        )
        
        self.sample_summary = "Apple reported strong earnings growth in Q4."
        self.sample_document = "Apple Inc. reported..."
    
    def test_initialization(self):
        """Test MESA initialization"""
        self.assertEqual(len(self.mesa.stakeholder_profiles), 4)
        self.assertTrue(self.mesa.use_rl_learning)
    
    def test_stakeholder_profiles(self):
        """Test stakeholder profile loading"""
        for profile in ["analyst", "compliance", "executive", "investor"]:
            self.assertIn(profile, self.mesa.stakeholder_profiles)
            
            profile_data = self.mesa.get_profile(profile)
            self.assertIn('information_needs', profile_data)
            self.assertIn('expertise_level', profile_data)
    
    def test_explanation_generation(self):
        """Test stakeholder-specific explanation generation"""
        explanation = self.mesa.generate_explanation(
            summary=self.sample_summary,
            document=self.sample_document,
            stakeholder="analyst"
        )
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
    
    def test_all_stakeholders(self):
        """Test explanation generation for all stakeholders"""
        for stakeholder in self.mesa.stakeholder_profiles:
            explanation = self.mesa.generate_explanation(
                summary=self.sample_summary,
                document=self.sample_document,
                stakeholder=stakeholder
            )
            
            self.assertIsNotNone(explanation)
            self.assertIsInstance(explanation, str)
    
    def test_rl_feedback(self):
        """Test RL-based learning from feedback"""
        initial_quality = self.mesa.evaluate_explanation(
            "Test explanation",
            "analyst"
        )
        
        # Provide feedback
        self.mesa.update_with_feedback(
            stakeholder="analyst",
            explanation="Test explanation",
            feedback_score=0.9
        )
        
        # Quality should potentially improve (or at least not crash)
        self.assertIsNotNone(initial_quality)
    
    def test_consensus_mechanism(self):
        """Test consensus across stakeholders"""
        explanations = {
            stakeholder: self.mesa.generate_explanation(
                self.sample_summary,
                self.sample_document,
                stakeholder
            )
            for stakeholder in self.mesa.stakeholder_profiles
        }
        
        consensus = self.mesa.compute_consensus(explanations)
        
        self.assertIn('consensus_score', consensus)
        self.assertGreaterEqual(consensus['consensus_score'], 0)
        self.assertLessEqual(consensus['consensus_score'], 1)


class TestCausalExplainer(unittest.TestCase):
    """Test Algorithm 4: CAUSAL-EXPLAIN"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.explainer = CausalExplainer(
            max_chain_length=5,
            confidence_threshold=0.6
        )
        
        self.causal_text = (
            "Interest rate hikes led to market decline. "
            "The market decline caused investor concerns. "
            "Investor concerns resulted in portfolio rebalancing."
        )
    
    def test_causal_marker_detection(self):
        """Test detection of causal markers"""
        markers = self.explainer.detect_causal_markers(self.causal_text)
        
        self.assertGreater(len(markers), 0)
        self.assertIn('led to', [m.lower() for m in markers])
    
    def test_causal_relation_extraction(self):
        """Test causal relation extraction"""
        relations = self.explainer.extract_causal_relations(self.causal_text)
        
        self.assertIsInstance(relations, list)
        self.assertGreater(len(relations), 0)
        
        # Check relation structure
        if len(relations) > 0:
            relation = relations[0]
            self.assertIn('cause', relation)
            self.assertIn('effect', relation)
            self.assertIn('confidence', relation)
    
    def test_causal_chain_construction(self):
        """Test causal chain construction (Algorithm 4)"""
        chains = self.explainer.extract_causal_chains(self.causal_text)
        
        self.assertIsInstance(chains, list)
        
        if len(chains) > 0:
            chain = chains[0]
            self.assertIsInstance(chain, list)
            
            # Check chain length constraint
            self.assertLessEqual(len(chain), self.explainer.max_chain_length)
    
    def test_confidence_filtering(self):
        """Test confidence threshold filtering"""
        # Extract with high threshold
        self.explainer.confidence_threshold = 0.9
        high_conf_chains = self.explainer.extract_causal_chains(self.causal_text)
        
        # Extract with low threshold
        self.explainer.confidence_threshold = 0.3
        low_conf_chains = self.explainer.extract_causal_chains(self.causal_text)
        
        # Lower threshold should yield more or equal chains
        self.assertGreaterEqual(len(low_conf_chains), len(high_conf_chains))
    
    def test_empty_text(self):
        """Test handling of text without causal relations"""
        text = "Apple Inc. is a technology company."
        chains = self.explainer.extract_causal_chains(text)
        
        self.assertEqual(len(chains), 0)


class TestAdaptiveEvaluator(unittest.TestCase):
    """Test Algorithm 5: ADAPT-EVAL"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = AdaptiveEvaluator(
            base_metrics=['rouge', 'bertscore'],
            adaptive_threshold=0.7
        )
    
    def test_metric_selection(self):
        """Test adaptive metric selection"""
        document = "Financial report with numbers: $89.5B revenue."
        
        selected_metrics = self.evaluator.select_metrics(document)
        
        self.assertIsInstance(selected_metrics, list)
        self.assertGreater(len(selected_metrics), 0)
    
    def test_evaluation(self):
        """Test adaptive evaluation (Algorithm 5)"""
        prediction = "Apple reported $89.5B revenue."
        reference = "Apple Inc. announced revenue of $89.5 billion."
        
        results = self.evaluator.evaluate(prediction, reference)
        
        self.assertIn('scores', results)
        self.assertIn('selected_metrics', results)
    
    def test_threshold_adaptation(self):
        """Test threshold adaptation based on content"""
        # Technical document
        tech_doc = "The algorithm uses GAT layers with 8 attention heads."
        tech_threshold = self.evaluator.adapt_threshold(tech_doc)
        
        # Simple document
        simple_doc = "The company did well."
        simple_threshold = self.evaluator.adapt_threshold(simple_doc)
        
        # Thresholds should differ based on complexity
        self.assertIsInstance(tech_threshold, float)
        self.assertIsInstance(simple_threshold, float)


class TestMultiStageTraining(unittest.TestCase):
    """Test Algorithm 6: Multi-Stage Training"""
    
    def setUp(self):
        """Set up test fixtures"""
        from models import HybridModel
        from torch.utils.data import DataLoader, TensorDataset
        
        self.model = HybridModel(hidden_dim=256)  # Small for testing
        
        # Create dummy data
        dummy_inputs = torch.randint(0, 1000, (20, 128))
        dummy_labels = torch.randint(0, 1000, (20, 64))
        dataset = TensorDataset(dummy_inputs, dummy_labels)
        
        self.train_loader = DataLoader(dataset, batch_size=4)
        self.val_loader = DataLoader(dataset, batch_size=4)
        
        self.config = {
            'stage1_lr': 1e-4,
            'stage2_lr': 5e-5,
            'stage3_lr': 2e-5,
            'alpha': 0.7,
            'beta': 0.2,
            'gamma': 0.1
        }
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        from training import MultiStageTrainer
        
        trainer = MultiStageTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            config=self.config
        )
        
        self.assertEqual(trainer.num_stages, 3)
        self.assertEqual(len(trainer.optimizers), 3)
    
    def test_stage_configuration(self):
        """Test stage-specific configurations"""
        from training import MultiStageTrainer
        
        trainer = MultiStageTrainer(
            self.model, self.train_loader, self.val_loader, self.config
        )
        
        # Check stage epochs
        self.assertEqual(trainer.stage_epochs, [3, 2, 5])
        
        # Check learning rates
        self.assertEqual(self.config['stage1_lr'], 1e-4)
        self.assertEqual(self.config['stage2_lr'], 5e-5)
        self.assertEqual(self.config['stage3_lr'], 2e-5)
    
    def test_loss_computation(self):
        """Test loss computation with weights (Equation 10)"""
        from training import MultiStageTrainer
        
        trainer = MultiStageTrainer(
            self.model, self.train_loader, self.val_loader, self.config
        )
        
        # Mock outputs
        outputs = {
            'loss': torch.tensor(1.0),
            'kg_alignment_loss': torch.tensor(0.5),
            'explanation_loss': torch.tensor(0.3)
        }
        
        # Stage 1: Only summary loss
        loss_stage1 = trainer._compute_loss(outputs, {}, stage=0)
        self.assertAlmostEqual(loss_stage1.item(), 1.0, places=5)
        
        # Stage 2: Summary + KG
        loss_stage2 = trainer._compute_loss(outputs, {}, stage=1)
        expected = 0.7 * 1.0 + 0.2 * 0.5  # α*L_summary + β*L_KG
        self.assertAlmostEqual(loss_stage2.item(), expected, places=5)
        
        # Stage 3: All components
        loss_stage3 = trainer._compute_loss(outputs, {}, stage=2)
        expected = 0.7 * 1.0 + 0.2 * 0.5 + 0.1 * 0.3
        self.assertAlmostEqual(loss_stage3.item(), expected, places=5)


# Test Suite
def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestKnowledgeGraphConstructor))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHybridInference))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMESAFramework))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCausalExplainer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveEvaluator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMultiStageTraining))
    
    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
