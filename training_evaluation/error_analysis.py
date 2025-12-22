"""
Error Analysis
==============

Comprehensive error categorization and analysis following Section V.D
from the paper.

Error Categories:
1. Entity Misidentification (21.8%)
2. Causal Misattribution (15.2%)
3. Temporal Inconsistency (12.5%)
4. Factual Errors (7.0%)
5. Other (43.5%)

Reference: Section V.D (Error Analysis)
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Detailed error analysis and categorization
    
    Analyzes prediction errors and categorizes them by type,
    severity, and patterns.
    """
    
    def __init__(self):
        """Initialize error analyzer"""
        # Error categories from paper (Section V.D)
        self.error_categories = {
            'entity_misidentification': {
                'name': 'Entity Misidentification',
                'description': 'Incorrect entity extraction or linking',
                'expected_rate': 0.218
            },
            'causal_misattribution': {
                'name': 'Causal Misattribution',
                'description': 'Incorrect causal relationships',
                'expected_rate': 0.152
            },
            'temporal_inconsistency': {
                'name': 'Temporal Inconsistency',
                'description': 'Timeline or temporal ordering errors',
                'expected_rate': 0.125
            },
            'factual_error': {
                'name': 'Factual Error',
                'description': 'Incorrect facts or hallucinations',
                'expected_rate': 0.070
            },
            'other': {
                'name': 'Other Errors',
                'description': 'Miscellaneous errors',
                'expected_rate': 0.435
            }
        }
        
        # Severity levels
        self.severity_levels = {
            'low': (0, 33),      # 0-33%: Minor issues
            'medium': (34, 66),  # 34-66%: Moderate issues
            'high': (67, 100)    # 67-100%: Critical issues
        }
        
        logger.info("Error Analyzer initialized")
    
    def analyze(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Perform comprehensive error analysis
        
        Args:
            predictions: Model predictions
            references: Reference summaries
            sources: Source documents
            metadata: Optional metadata per sample
            
        Returns:
            Detailed error analysis results
        """
        logger.info("=" * 70)
        logger.info("Starting Error Analysis")
        logger.info("=" * 70)
        
        errors_by_category = defaultdict(list)
        errors_by_severity = defaultdict(list)
        detailed_errors = []
        
        for i, (pred, ref, src) in enumerate(zip(predictions, references, sources)):
            # Analyze this sample
            sample_errors = self._analyze_sample(pred, ref, src, i)
            
            # Categorize
            for error in sample_errors:
                errors_by_category[error['category']].append(error)
                errors_by_severity[error['severity']].append(error)
                detailed_errors.append(error)
        
        # Compute statistics
        stats = self._compute_statistics(
            errors_by_category,
            errors_by_severity,
            len(predictions)
        )
        
        # Print summary
        self._print_summary(stats)
        
        return {
            'statistics': stats,
            'errors_by_category': errors_by_category,
            'errors_by_severity': errors_by_severity,
            'detailed_errors': detailed_errors
        }
    
    def _analyze_sample(
        self,
        prediction: str,
        reference: str,
        source: str,
        sample_id: int
    ) -> List[Dict]:
        """Analyze errors in a single sample"""
        errors = []
        
        # Check for entity misidentification
        entity_errors = self._check_entity_errors(prediction, reference, source)
        errors.extend(entity_errors)
        
        # Check for causal misattribution
        causal_errors = self._check_causal_errors(prediction, reference)
        errors.extend(causal_errors)
        
        # Check for temporal inconsistency
        temporal_errors = self._check_temporal_errors(prediction, reference)
        errors.extend(temporal_errors)
        
        # Check for factual errors
        factual_errors = self._check_factual_errors(prediction, source)
        errors.extend(factual_errors)
        
        # Add sample ID to all errors
        for error in errors:
            error['sample_id'] = sample_id
        
        return errors
    
    def _check_entity_errors(
        self,
        prediction: str,
        reference: str,
        source: str
    ) -> List[Dict]:
        """Check for entity misidentification errors"""
        errors = []
        
        # Extract entities (simplified)
        pred_entities = self._extract_entities(prediction)
        ref_entities = self._extract_entities(reference)
        source_entities = self._extract_entities(source)
        
        # Check for hallucinated entities
        for entity in pred_entities:
            if entity not in source_entities:
                errors.append({
                    'category': 'entity_misidentification',
                    'type': 'hallucinated_entity',
                    'entity': entity,
                    'severity': self._compute_severity(0.8),
                    'description': f"Entity '{entity}' not in source"
                })
        
        # Check for missing entities
        for entity in ref_entities:
            if entity not in pred_entities:
                errors.append({
                    'category': 'entity_misidentification',
                    'type': 'missing_entity',
                    'entity': entity,
                    'severity': self._compute_severity(0.6),
                    'description': f"Entity '{entity}' missing from prediction"
                })
        
        return errors
    
    def _check_causal_errors(
        self,
        prediction: str,
        reference: str
    ) -> List[Dict]:
        """Check for causal misattribution errors"""
        errors = []
        
        # Extract causal patterns
        causal_markers = ['led to', 'caused', 'resulted in', 'due to', 'because of']
        
        pred_causals = []
        ref_causals = []
        
        for marker in causal_markers:
            # Find causal relationships in prediction
            if marker in prediction.lower():
                context = self._extract_context(prediction, marker)
                pred_causals.append((marker, context))
            
            # Find causal relationships in reference
            if marker in reference.lower():
                context = self._extract_context(reference, marker)
                ref_causals.append((marker, context))
        
        # Check for incorrect causal attributions
        if pred_causals and not any(
            self._causal_matches(pc, ref_causals)
            for pc in pred_causals
        ):
            errors.append({
                'category': 'causal_misattribution',
                'type': 'incorrect_causation',
                'severity': self._compute_severity(0.7),
                'description': 'Causal relationship not supported'
            })
        
        return errors
    
    def _check_temporal_errors(
        self,
        prediction: str,
        reference: str
    ) -> List[Dict]:
        """Check for temporal inconsistency errors"""
        errors = []
        
        # Extract temporal expressions
        temporal_pattern = r'\b(Q[1-4]\s+\d{4}|\d{4}|yesterday|today|last\s+\w+)\b'
        
        pred_temporal = re.findall(temporal_pattern, prediction, re.IGNORECASE)
        ref_temporal = re.findall(temporal_pattern, reference, re.IGNORECASE)
        
        # Check for temporal inconsistencies
        if pred_temporal and ref_temporal:
            if not any(pt in ref_temporal for pt in pred_temporal):
                errors.append({
                    'category': 'temporal_inconsistency',
                    'type': 'wrong_timeframe',
                    'severity': self._compute_severity(0.6),
                    'description': 'Temporal expressions do not match'
                })
        
        # Check for temporal ordering issues
        if len(pred_temporal) > 1:
            # Simplified check: ensure chronological order
            # (In production, would parse and validate dates)
            pass
        
        return errors
    
    def _check_factual_errors(
        self,
        prediction: str,
        source: str
    ) -> List[Dict]:
        """Check for factual errors"""
        errors = []
        
        # Extract numerical facts
        pred_numbers = re.findall(r'\$?[\d,]+\.?\d*[BMK]?', prediction)
        source_numbers = re.findall(r'\$?[\d,]+\.?\d*[BMK]?', source)
        
        # Check for hallucinated numbers
        for num in pred_numbers:
            if num not in source_numbers:
                # Check if similar number exists (allow small variations)
                if not self._number_exists_similar(num, source_numbers):
                    errors.append({
                        'category': 'factual_error',
                        'type': 'hallucinated_number',
                        'value': num,
                        'severity': self._compute_severity(0.9),
                        'description': f"Number '{num}' not in source"
                    })
        
        return errors
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified)"""
        # Simplified: extract capitalized phrases
        entities = []
        
        # Company names (capitalized + Inc/Corp/etc)
        companies = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|Corp\.|Ltd\.)',
            text
        )
        entities.extend(companies)
        
        # Standalone capitalized words (likely entities)
        caps = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(caps)
        
        return list(set(entities))
    
    def _extract_context(self, text: str, marker: str, window: int = 50) -> str:
        """Extract context around marker"""
        idx = text.lower().find(marker.lower())
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(marker) + window)
        
        return text[start:end]
    
    def _causal_matches(
        self,
        pred_causal: Tuple,
        ref_causals: List[Tuple]
    ) -> bool:
        """Check if predicted causal matches any reference"""
        pred_marker, pred_context = pred_causal
        
        for ref_marker, ref_context in ref_causals:
            # Simple overlap check
            pred_words = set(pred_context.lower().split())
            ref_words = set(ref_context.lower().split())
            
            overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
            if overlap > 0.3:
                return True
        
        return False
    
    def _number_exists_similar(
        self,
        target: str,
        numbers: List[str],
        threshold: float = 0.05
    ) -> bool:
        """Check if similar number exists"""
        # Extract numeric value
        target_val = self._parse_number(target)
        if target_val is None:
            return False
        
        for num in numbers:
            num_val = self._parse_number(num)
            if num_val is None:
                continue
            
            # Check if within threshold
            if abs(target_val - num_val) / num_val < threshold:
                return True
        
        return False
    
    def _parse_number(self, num_str: str) -> Optional[float]:
        """Parse number string to float"""
        try:
            # Remove commas and currency symbols
            clean = re.sub(r'[$,]', '', num_str)
            
            # Handle suffixes
            if clean.endswith('B'):
                return float(clean[:-1]) * 1e9
            elif clean.endswith('M'):
                return float(clean[:-1]) * 1e6
            elif clean.endswith('K'):
                return float(clean[:-1]) * 1e3
            else:
                return float(clean)
        except:
            return None
    
    def _compute_severity(self, error_score: float) -> str:
        """Compute error severity level"""
        percentage = error_score * 100
        
        for severity, (low, high) in self.severity_levels.items():
            if low <= percentage <= high:
                return severity
        
        return 'medium'
    
    def _compute_statistics(
        self,
        errors_by_category: Dict,
        errors_by_severity: Dict,
        total_samples: int
    ) -> Dict:
        """Compute error statistics"""
        stats = {
            'total_samples': total_samples,
            'total_errors': sum(len(errors) for errors in errors_by_category.values()),
            'error_rate': 0.0,
            'category_distribution': {},
            'severity_distribution': {},
            'comparison_to_expected': {}
        }
        
        # Error rate
        stats['error_rate'] = stats['total_errors'] / total_samples if total_samples > 0 else 0
        
        # Category distribution
        for category, errors in errors_by_category.items():
            count = len(errors)
            percentage = count / total_samples * 100 if total_samples > 0 else 0
            
            stats['category_distribution'][category] = {
                'count': count,
                'percentage': percentage
            }
            
            # Compare to expected (from paper)
            expected = self.error_categories[category]['expected_rate'] * 100
            stats['comparison_to_expected'][category] = {
                'observed': percentage,
                'expected': expected,
                'difference': percentage - expected
            }
        
        # Severity distribution
        for severity, errors in errors_by_severity.items():
            count = len(errors)
            percentage = count / stats['total_errors'] * 100 if stats['total_errors'] > 0 else 0
            
            stats['severity_distribution'][severity] = {
                'count': count,
                'percentage': percentage
            }
        
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print error analysis summary"""
        logger.info("\n" + "=" * 70)
        logger.info("ERROR ANALYSIS SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"\nTotal Samples: {stats['total_samples']}")
        logger.info(f"Total Errors: {stats['total_errors']}")
        logger.info(f"Error Rate: {stats['error_rate']:.2%}")
        
        logger.info("\nError Distribution by Category:")
        for category, data in stats['category_distribution'].items():
            logger.info(f"  {category}: {data['count']} ({data['percentage']:.1f}%)")
        
        logger.info("\nSeverity Distribution:")
        for severity, data in stats['severity_distribution'].items():
            logger.info(f"  {severity}: {data['count']} ({data['percentage']:.1f}%)")
        
        logger.info("\nComparison to Expected (from Paper):")
        for category, data in stats['comparison_to_expected'].items():
            diff = data['difference']
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            logger.info(
                f"  {category}: {data['observed']:.1f}% "
                f"(expected: {data['expected']:.1f}%) {direction}"
            )


# Example usage
if __name__ == "__main__":
    print("Error Analysis Module")
    print("=" * 70)
    
    print("\nError Categories (from Section V.D):")
    print("  1. Entity Misidentification: 21.8%")
    print("  2. Causal Misattribution: 15.2%")
    print("  3. Temporal Inconsistency: 12.5%")
    print("  4. Factual Errors: 7.0%")
    print("  5. Other: 43.5%")
    
    print("\nSeverity Levels:")
    print("  - Low (0-33%): Minor issues")
    print("  - Medium (34-66%): Moderate issues")
    print("  - High (67-100%): Critical issues")
    
    print("\nAnalysis Features:")
    print("  ✓ Automatic error categorization")
    print("  ✓ Severity assessment")
    print("  ✓ Comparison to paper baselines")
    print("  ✓ Detailed error tracking")
    
    print("\n" + "=" * 70)
    print("Error analyzer ready!")
    print("=" * 70)
