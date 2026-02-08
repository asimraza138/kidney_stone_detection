"""
Training script for kidney stone detection model.
Implements training loop with validation, checkpointing, and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings

from models.architecture import build_model
from models.losses import MultiTaskLoss
from utils.metrics import MetricsTracker
from utils.config import Config


class Trainer:
    """
    Trainer class for kidney stone detection model.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Config):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        self.device = config.get_device()
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
        
        # Loss function
        self.criterion = MultiTaskLoss(config)
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = config.CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_optimizer(self):
        """Build optimizer based on config"""
        if self.config.OPTIMIZER == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'sgd':
            optimizer = SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.LR_SCHEDULER == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
                eta_min=self.config.MIN_LR
            )
        elif self.config.LR_SCHEDULER == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.LR_SCHEDULER == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self) -> dict:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with average loss values
        """
        self.model.train()
        
        epoch_losses = {
            'total': [],
            'classification': [],
            'regression': []
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            sizes = batch['size'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    
                    targets = {
                        'label': labels,
                        'size': sizes
                    }
                    
                    losses = self.criterion(outputs, targets)
                    loss = losses['total']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.CLIP_GRAD_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.CLIP_GRAD_NORM
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(images)
                
                targets = {
                    'label': labels,
                    'size': sizes
                }
                
                losses = self.criterion(outputs, targets)
                loss = losses['total']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.CLIP_GRAD_NORM
                    )
                
                self.optimizer.step()
            
            # Record losses
            epoch_losses['total'].append(loss.item())
            epoch_losses['classification'].append(losses['classification'].item())
            epoch_losses['regression'].append(losses['regression'].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self) -> tuple:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        self.metrics_tracker.reset()
        
        epoch_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                sizes = batch['size'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Get predictions
                predictions = self.model.predict(images, 
                                                 threshold=self.config.CONFIDENCE_THRESHOLD)
                
                # Calculate loss
                targets = {
                    'label': labels,
                    'size': sizes
                }
                
                losses = self.criterion(outputs, targets)
                epoch_losses.append(losses['total'].item())
                
                # Update metrics
                self.metrics_tracker.update(predictions, targets)
        
        # Compute metrics
        metrics = self.metrics_tracker.compute_all()
        avg_loss = np.mean(epoch_losses)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            print(f"\n{'='*70}")
            print(f"Epoch {self.current_epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Print epoch summary
            print(f"\nTraining Loss: {train_losses['total']:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation F1: {val_metrics['detection']['f1_score']:.4f}")
            print(f"Validation MAE: {val_metrics['regression']['mae']:.4f} mm")
            
            # Check for improvement
            val_f1 = val_metrics['detection']['f1_score']
            is_best = val_f1 > self.best_val_f1
            
            if is_best:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                print(f"*** New best model! F1: {val_f1:.4f} ***")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.PATIENCE:
                print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                print(f"No improvement for {self.config.PATIENCE} consecutive epochs")
                break
        
        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*70)
        
        # Save training history
        self._save_history()
    
    def _save_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / "training_history.json"
        
        # Convert to serializable format
        history_serializable = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'val_metrics': [
                {k: v if not isinstance(v, dict) else str(v) 
                 for k, v in metrics.items()}
                for metrics in self.history['val_metrics']
            ]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved to {history_path}")


def main():
    """Main training function"""
    # Load configuration
    config = Config()
    config.print_config()
    
    # Import here to avoid circular imports
    from data.dataset import create_dataloaders
    
    # Create dataloaders
    # NOTE: Update these paths to your actual annotation files
    train_loader, val_loader, _ = create_dataloaders(
        config,
        train_annotations='annotations/train.csv',
        val_annotations='annotations/val.csv'
    )
    
    # Build model
    model = build_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
