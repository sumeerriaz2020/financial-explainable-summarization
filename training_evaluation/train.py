"""
Main Training Script
====================

Implements multi-stage training pipeline (Algorithm 6) for hybrid
neural-symbolic model with knowledge graph integration.

Reference: Algorithm 6 (Multi-Stage Training)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiStageTrainer:
    """
    Multi-Stage Training Pipeline (Algorithm 6)
    
    Stage 1: Warm-up phase (3 epochs)
    Stage 2: Knowledge integration (2 epochs)
    Stage 3: End-to-end fine-tuning (5 epochs)
    """
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize multi-stage trainer
        
        Args:
            model: HybridModel instance
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            device: Computing device
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # Training stages
        self.num_stages = 3
        self.stage_epochs = [3, 2, 5]  # Algorithm 6
        
        # Loss weights (Equation 10)
        self.alpha = config.get('alpha', 0.7)  # Summary loss weight
        self.beta = config.get('beta', 0.2)   # KG alignment weight
        self.gamma = config.get('gamma', 0.1)  # Explanation loss weight
        
        # Optimizers for each stage
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'stage_metrics': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        logger.info(f"Multi-Stage Trainer initialized on {device}")
        logger.info(f"Stages: {self.stage_epochs}")
    
    def _setup_optimizers(self) -> List[torch.optim.Optimizer]:
        """Setup optimizers for each stage"""
        optimizers = []
        
        # Stage 1: Warm-up - only decoder
        stage1_params = list(self.model.decoder.parameters())
        optimizers.append(
            AdamW(stage1_params, lr=self.config.get('stage1_lr', 1e-4))
        )
        
        # Stage 2: KG integration - encoder + KG encoder
        stage2_params = (
            list(self.model.text_encoder.parameters()) +
            list(self.model.kg_encoder.parameters()) +
            list(self.model.cross_attention.parameters())
        )
        optimizers.append(
            AdamW(stage2_params, lr=self.config.get('stage2_lr', 5e-5))
        )
        
        # Stage 3: End-to-end - all parameters
        stage3_params = self.model.parameters()
        optimizers.append(
            AdamW(stage3_params, lr=self.config.get('stage3_lr', 2e-5))
        )
        
        return optimizers
    
    def _setup_schedulers(self) -> List:
        """Setup learning rate schedulers"""
        schedulers = []
        
        for i, optimizer in enumerate(self.optimizers):
            num_epochs = self.stage_epochs[i]
            num_steps = len(self.train_dataloader) * num_epochs
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * num_steps),
                num_training_steps=num_steps
            )
            schedulers.append(scheduler)
        
        return schedulers
    
    def train(self, output_dir: str):
        """
        Execute complete multi-stage training
        
        Args:
            output_dir: Directory to save checkpoints
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("Starting Multi-Stage Training")
        logger.info("=" * 70)
        
        total_epochs = sum(self.stage_epochs)
        current_epoch = 0
        
        for stage in range(self.num_stages):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"STAGE {stage + 1}: {self._get_stage_name(stage)}")
            logger.info(f"{'=' * 70}")
            
            stage_epochs = self.stage_epochs[stage]
            optimizer = self.optimizers[stage]
            scheduler = self.schedulers[stage]
            
            for epoch in range(stage_epochs):
                current_epoch += 1
                
                logger.info(f"\nEpoch {current_epoch}/{total_epochs}")
                
                # Training
                train_loss = self._train_epoch(
                    stage, optimizer, scheduler
                )
                
                # Validation
                val_loss, val_metrics = self._validate_epoch()
                
                # Record history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                
                logger.info(f"Train Loss: {train_loss:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}")
                
                # Save checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_path = self._save_checkpoint(
                        output_dir / f'best_model_stage{stage+1}.pt',
                        stage, current_epoch
                    )
                    logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save stage checkpoint
            stage_path = self._save_checkpoint(
                output_dir / f'stage{stage+1}_final.pt',
                stage, current_epoch
            )
            
            logger.info(f"\n✓ Stage {stage + 1} completed")
            logger.info(f"Checkpoint saved: {stage_path}")
        
        # Save training history
        self._save_history(output_dir / 'training_history.json')
        
        logger.info(f"\n{'=' * 70}")
        logger.info("Training Complete!")
        logger.info(f"Best model: {self.best_model_path}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"{'=' * 70}")
    
    def _train_epoch(
        self,
        stage: int,
        optimizer: torch.optim.Optimizer,
        scheduler
    ) -> float:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                kg_node_features=batch.get('kg_features'),
                kg_adjacency=batch.get('kg_adj'),
                labels=batch['labels']
            )
            
            # Compute loss (Equation 10)
            loss = self._compute_loss(outputs, batch, stage)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> tuple:
        """Validate one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    kg_node_features=batch.get('kg_features'),
                    kg_adjacency=batch.get('kg_adj'),
                    labels=batch['labels']
                )
                
                # Compute loss
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics = {}  # Additional metrics can be computed here
        
        return avg_loss, metrics
    
    def _compute_loss(
        self,
        outputs: Dict,
        batch: Dict,
        stage: int
    ) -> torch.Tensor:
        """
        Compute multi-component loss (Equation 10)
        
        L_total = α*L_summary + β*L_KG + γ*L_explanation
        """
        # Summary loss (cross-entropy)
        loss_summary = outputs['loss']
        
        # Stage-specific loss components
        if stage == 0:
            # Stage 1: Focus on summary only
            total_loss = loss_summary
        
        elif stage == 1:
            # Stage 2: Add KG alignment
            loss_kg = outputs.get('kg_alignment_loss', 0.0)
            total_loss = self.alpha * loss_summary + self.beta * loss_kg
        
        else:
            # Stage 3: All components
            loss_kg = outputs.get('kg_alignment_loss', 0.0)
            loss_explanation = outputs.get('explanation_loss', 0.0)
            
            total_loss = (
                self.alpha * loss_summary +
                self.beta * loss_kg +
                self.gamma * loss_explanation
            )
        
        return total_loss
    
    def _save_checkpoint(
        self,
        path: Path,
        stage: int,
        epoch: int
    ) -> Path:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, path)
        return path
    
    def _save_history(self, path: Path):
        """Save training history"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _get_stage_name(self, stage: int) -> str:
        """Get stage name"""
        names = [
            "Warm-up Phase",
            "Knowledge Integration",
            "End-to-End Fine-tuning"
        ]
        return names[stage]


def load_checkpoint(path: str, model, device: str = 'cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


# Example usage
if __name__ == "__main__":
    print("Multi-Stage Training Script")
    print("=" * 70)
    
    # Configuration
    config = {
        'batch_size': 16,
        'stage1_lr': 1e-4,
        'stage2_lr': 5e-5,
        'stage3_lr': 2e-5,
        'alpha': 0.7,
        'beta': 0.2,
        'gamma': 0.1,
        'max_grad_norm': 1.0
    }
    
    # Initialize (mock data)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Stages:")
    print("  Stage 1: Warm-up (3 epochs)")
    print("  Stage 2: Knowledge Integration (2 epochs)")
    print("  Stage 3: End-to-End Fine-tuning (5 epochs)")
    print("  Total: 10 epochs")
    
    print("\nLoss Function (Equation 10):")
    print("  L_total = α*L_summary + β*L_KG + γ*L_explanation")
    print(f"  α={config['alpha']}, β={config['beta']}, γ={config['gamma']}")
    
    print("\n" + "=" * 70)
    print("Ready to train!")
    print("=" * 70)
