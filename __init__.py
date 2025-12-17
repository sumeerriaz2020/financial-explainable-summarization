"""
Financial Explainable Summarization
====================================

Hybrid Neural-Symbolic Framework for Explainable Financial Document Summarization

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
Affiliation: IQRA University Islamabad Campus
"""

__version__ = "1.0.0"
__author__ = "Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi"
__email__ = "sumeer33885@iqraisb.edu.pk"
__license__ = "MIT"

# Import main components for easy access
from .models import HybridModel, DualEncoder, KnowledgeGraphEncoder
from .knowledge_graph import FIBOIntegration, EntityLinker, CausalExtractor
from .explainability import MESAExplainer, CausalExplainer, TemporalExplainer
from .utils import DocumentPreprocessor, FinancialNER, KnowledgeGraphUtils

__all__ = [
    # Models
    'HybridModel',
    'DualEncoder', 
    'KnowledgeGraphEncoder',
    
    # Knowledge Graph
    'FIBOIntegration',
    'EntityLinker',
    'CausalExtractor',
    
    # Explainability
    'MESAExplainer',
    'CausalExplainer',
    'TemporalExplainer',
    
    # Utils
    'DocumentPreprocessor',
    'FinancialNER',
    'KnowledgeGraphUtils',
]
