"""
Algorithm 6: Multi-Stage Training with Joint Optimization
===========================================================

Implementation of Algorithm 6 from the paper.

Multi-stage training procedure that progressively enhances model capabilities:
Stage 1: Domain pre-training on financial corpora
Stage 2: Knowledge-aware attention learning with FIBO integration
Stage 3: Joint optimization of summarization and explainability

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
Reference: Section III.F.1, Algorithm 6
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for multi-stage training"""
    # Stage 1: Domain pre-training
    stage1_lr: float = 5e-5
    stage1_epochs: int = 3
    stage1_batch_size: int = 16
    
    # Stage 2: Knowledge integration
    stage2_lr: float = 3e-5
    stage2_epochs: int = 2
    stage2_batch_size: int = 16
    kg_attention_weight: float = 0.5  # λ in Equation 3
    
    # Stage 3: Joint optimization
    stage3_lr: float = 2e-5
    stage3_epochs: int = 5
    stage3_batch_size: int = 16
    patience: int = 5  # Early stopping
    
    # Loss weights (Equation 10)
    alpha: float = 0.35  # Summarization loss
    beta: float = 0.25   # Explanation loss
    gamma: float = 0.20  # Causal loss
    delta: float = 0.10  # Consistency loss
    epsilon: float = 0.10  # Stakeholder loss
    
    # Optimization
    gradient_clip: float = 1.0
    warmup_steps: int = 500
    lr_decay: float = 0.9
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FinancialDocumentDataset(Dataset):
    """Dataset for financial documents with annotations"""
    
    def __init__(
        self,
        documents: List[Dict],
        tokenizer: any,
        max_length: int = 1024
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            doc['text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize reference summary
        if 'summary' in doc:
            labels = self.tokenizer(
                doc['summary'],
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
        else:
            labels = None
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0) if labels else None,
            'stakeholder': doc.get('stakeholder', 'analyst'),
            'causal_chains': doc.get('causal_chains', []),
            'kg_local': doc.get('kg_local', None)
        }


class MultiStageTrainer:
    """
    Multi-stage training orchestrator for hybrid neural-symbolic model.
    
    Implements Algorithm 6 for progressive model enhancement through
    three training stages with composite loss optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        config: TrainingConfig
    ):
        """
        Initialize multi-stage trainer
        
        Args:
            model: Hybrid neural-symbolic model
            train_data: Training data loader
            val_data: Validation data loader
            config: Training configuration
        """
        self.model = model.to(config.device)
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        
        # Best model tracking
        self.best_val_perf = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on {config.device}")
        logger.info(f"Training samples: {len(train_data.dataset)}")
        logger.info(f"Validation samples: {len(val_data.dataset)}")
    
    def train_all_stages(self) -> Dict[str, any]:
        """
        Execute complete multi-stage training pipeline.
        
        Implements Algorithm 6 from paper (Section III.F.1).
        
        Returns:
            Dictionary with training history and final model
        """
        logger.info("="*70)
        logger.info("MULTI-STAGE TRAINING PIPELINE")
        logger.info("="*70)
        
        history = {
            'stage1': {},
            'stage2': {},
            'stage3': {}
        }
        
        # Stage 1: Domain Pre-training
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: Domain Pre-training")
        logger.info("="*70)
        history['stage1'] = self._stage1_domain_pretraining()
        
        # Stage 2: Knowledge Integration
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: Knowledge Integration")
        logger.info("="*70)
        history['stage2'] = self._stage2_knowledge_integration()
        
        # Stage 3: Joint Optimization
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: Joint Optimization")
        logger.info("="*70)
        history['stage3'] = self._stage3_joint_optimization()
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"\nLoaded best model (val performance: {self.best_val_perf:.4f})")
        
        return {
            'history': history,
            'best_val_performance': self.best_val_perf,
            'final_model': self.model
        }
    
    def _stage1_domain_pretraining(self) -> Dict:
        """
        Stage 1: Domain pre-training on financial corpora.
        
        Lines 2-5 of Algorithm 6.
        Fine-tune transformer on financial texts to adapt language
        representations to financial terminology and concepts.
        """
        logger.info(f"Learning rate: {self.config.stage1_lr}")
        logger.info(f"Epochs: {self.config.stage1_epochs}")
        logger.info(f"Batch size: {self.config.stage1_batch_size}")
        
        # Initialize optimizer (Line 2)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.stage1_lr,
            weight_decay=0.01
        )
        
        history = {'train_loss': [], 'val_loss': []}
        
        # Training loop (Lines 3-5)
        for epoch in range(self.config.stage1_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.stage1_epochs}")
            
            # Training
            train_loss = self._train_epoch_stage1(optimizer)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = self._validate_stage1()
            history['val_loss'].append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        return history
    
    def _stage2_knowledge_integration(self) -> Dict:
        """
        Stage 2: Knowledge-aware attention learning.
        
        Lines 6-10 of Algorithm 6.
        Train knowledge graph integration with FIBO-aligned attention.
        """
        logger.info(f"Learning rate: {self.config.stage2_lr}")
        logger.info(f"KG attention weight (λ): {self.config.kg_attention_weight}")
        logger.info(f"Epochs: {self.config.stage2_epochs}")
        
        # Initialize optimizer (Line 6)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.stage2_lr,
            weight_decay=0.01
        )
        
        history = {'train_loss': [], 'val_loss': [], 'kg_alignment': []}
        
        # Training loop (Lines 7-10)
        for epoch in range(self.config.stage2_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.stage2_epochs}")
            
            # Training with KG integration
            train_loss, kg_align = self._train_epoch_stage2(optimizer)
            history['train_loss'].append(train_loss)
            history['kg_alignment'].append(kg_align)
            
            # Validation
            val_loss = self._validate_stage2()
            history['val_loss'].append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | "
                       f"KG Alignment: {kg_align:.4f} | Val Loss: {val_loss:.4f}")
        
        return history
    
    def _stage3_joint_optimization(self) -> Dict:
        """
        Stage 3: Joint optimization of summarization and explainability.
        
        Lines 11-26 of Algorithm 6.
        Multi-task learning with composite loss function (Equation 10).
        """
        logger.info(f"Learning rate: {self.config.stage3_lr}")
        logger.info(f"Loss weights: α={self.config.alpha}, β={self.config.beta}, "
                   f"γ={self.config.gamma}, δ={self.config.delta}, ε={self.config.epsilon}")
        logger.info(f"Epochs: {self.config.stage3_epochs}")
        logger.info(f"Early stopping patience: {self.config.patience}")
        
        # Initialize optimizer (Line 11)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.stage3_lr,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.config.lr_decay
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'loss_components': [],
            'learning_rate': []
        }
        
        self.best_val_perf = 0.0
        self.patience_counter = 0
        
        # Training loop (Lines 12-25)
        for epoch in range(self.config.stage3_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.stage3_epochs}")
            
            # Training (Lines 13-17)
            train_loss, loss_components = self._train_epoch_stage3(optimizer)
            history['train_loss'].append(train_loss)
            history['loss_components'].append(loss_components)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Validation (Line 18)
            val_perf = self._validate_stage3()
            history['val_loss'].append(val_perf)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Performance: {val_perf:.4f}")
            logger.info(f"  L_sum: {loss_components['summarization']:.4f}, "
                       f"L_expl: {loss_components['explanation']:.4f}, "
                       f"L_causal: {loss_components['causal']:.4f}")
            
            # Early stopping check (Lines 19-24)
            if val_perf > self.best_val_perf:
                # Line 20: Update best performance
                self.best_val_perf = val_perf
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                logger.info(f"  ✓ New best model! (val_perf: {val_perf:.4f})")
            else:
                # Line 22: Increment patience
                self.patience_counter += 1
                logger.info(f"  No improvement (patience: {self.patience_counter}/{self.config.patience})")
                
                # Line 23: Early stopping
                if self.patience_counter >= self.config.patience:
                    logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
            
            # Line 25: Learning rate decay
            scheduler.step()
            logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        return history
    
    def _train_epoch_stage1(self, optimizer: optim.Optimizer) -> float:
        """Train one epoch for stage 1 (domain pre-training)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_data, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            # Forward pass (Line 4)
            optimizer.zero_grad()
            
            # Standard cross-entropy loss for summarization
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # Backward pass (Line 4)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def _train_epoch_stage2(
        self,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """
        Train one epoch for stage 2 (knowledge integration).
        
        Implements Equation 3: α_kg = softmax(score(H_text, H_kg) × λ × R)
        """
        self.model.train()
        total_loss = 0.0
        total_kg_alignment = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_data, desc="Training (KG)")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            # Forward pass with KG integration (Lines 8-9)
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # Compute KG alignment score
            kg_alignment = self._compute_kg_alignment_score(outputs)
            
            # Backward pass (Line 9)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            total_kg_alignment += kg_alignment
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'kg_align': f'{kg_alignment:.3f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_kg_alignment = total_kg_alignment / num_batches
        
        return avg_loss, avg_kg_alignment
    
    def _train_epoch_stage3(
        self,
        optimizer: optim.Optimizer
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train one epoch for stage 3 (joint optimization).
        
        Implements Equation 10: L_total = αL_sum + βL_expl + γL_causal + δL_cons + εL_stake
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {
            'summarization': 0.0,
            'explanation': 0.0,
            'causal': 0.0,
            'consistency': 0.0,
            'stakeholder': 0.0
        }
        num_batches = 0
        
        progress_bar = tqdm(self.train_data, desc="Training (Joint)")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            # Forward pass (Lines 14-15)
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute component losses (Line 15)
            L_sum = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            L_expl = self._compute_explanation_loss(outputs, batch)
            L_causal = self._compute_causal_loss(outputs, batch)
            L_cons = self._compute_consistency_loss(outputs, batch)
            L_stake = self._compute_stakeholder_loss(outputs, batch)
            
            # Composite loss (Equation 10, Line 15)
            L_total = (
                self.config.alpha * L_sum +
                self.config.beta * L_expl +
                self.config.gamma * L_causal +
                self.config.delta * L_cons +
                self.config.epsilon * L_stake
            )
            
            # Backward pass (Line 16)
            L_total.backward()
            
            # Gradient clipping (Line 16)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            # Update parameters (Line 16)
            optimizer.step()
            
            # Track losses
            total_loss += L_total.item()
            loss_components['summarization'] += L_sum.item()
            loss_components['explanation'] += L_expl.item()
            loss_components['causal'] += L_causal.item()
            loss_components['consistency'] += L_cons.item()
            loss_components['stakeholder'] += L_stake.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{L_total.item():.4f}',
                'grad': f'{grad_norm:.2f}'
            })
        
        # Average losses
        avg_total = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return avg_total, avg_components
    
    def _validate_stage1(self) -> float:
        """Validate stage 1 model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_data:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_stage2(self) -> float:
        """Validate stage 2 model"""
        return self._validate_stage1()  # Same validation for now
    
    def _validate_stage3(self) -> float:
        """Validate stage 3 model (returns performance metric, higher is better)"""
        self.model.eval()
        total_performance = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_data:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Compute performance metric (e.g., ROUGE-L or composite score)
                # For now, use negative loss as proxy
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                performance = -loss.item()  # Higher is better
                
                total_performance += performance
                num_batches += 1
        
        return total_performance / num_batches
    
    def _compute_kg_alignment_score(self, outputs: any) -> float:
        """Compute KG alignment score (mock implementation)"""
        # In production, check attention alignment with KG structure
        return np.random.uniform(0.7, 0.95)
    
    def _compute_explanation_loss(self, outputs: any, batch: Dict) -> torch.Tensor:
        """Compute explanation quality loss"""
        # Mock implementation - in production, measure explanation quality
        return torch.tensor(0.1, device=self.config.device)
    
    def _compute_causal_loss(self, outputs: any, batch: Dict) -> torch.Tensor:
        """Compute causal preservation loss"""
        # Mock implementation - measure causal chain preservation
        return torch.tensor(0.15, device=self.config.device)
    
    def _compute_consistency_loss(self, outputs: any, batch: Dict) -> torch.Tensor:
        """Compute temporal consistency loss"""
        # Mock implementation - measure consistency across time
        return torch.tensor(0.08, device=self.config.device)
    
    def _compute_stakeholder_loss(self, outputs: any, batch: Dict) -> torch.Tensor:
        """Compute stakeholder-specific loss"""
        # Mock implementation - measure stakeholder fit
        return torch.tensor(0.12, device=self.config.device)


# Example usage
if __name__ == "__main__":
    print("Multi-Stage Training Pipeline")
    print("="*70)
    
    # This is a demonstration - in production, use actual model and data
    print("\nInitializing training configuration...")
    
    config = TrainingConfig(
        stage1_epochs=2,
        stage2_epochs=2,
        stage3_epochs=3,
        device='cpu'  # Use CPU for demo
    )
    
    print(f"Device: {config.device}")
    print(f"Stage 1: {config.stage1_epochs} epochs, LR={config.stage1_lr}")
    print(f"Stage 2: {config.stage2_epochs} epochs, LR={config.stage2_lr}")
    print(f"Stage 3: {config.stage3_epochs} epochs, LR={config.stage3_lr}")
    print(f"\nLoss weights:")
    print(f"  α (summarization): {config.alpha}")
    print(f"  β (explanation): {config.beta}")
    print(f"  γ (causal): {config.gamma}")
    print(f"  δ (consistency): {config.delta}")
    print(f"  ε (stakeholder): {config.epsilon}")
    
    print("\nNote: This is a framework demonstration.")
    print("In production, initialize with actual model and datasets.")
    
    print(f"\n{'='*70}")
    print("Training pipeline ready!")
    print(f"{'='*70}")
