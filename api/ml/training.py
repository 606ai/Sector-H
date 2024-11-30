from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .experiment import ExperimentTracker
from ..config import get_settings
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training with experiment tracking."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        self.settings = get_settings()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.experiment_name = experiment_name or "default_experiment"
        
        # Initialize metrics
        self.best_metric: float = float('inf')
        self.best_model_path: Optional[Path] = None
        
        # Create output directory
        self.output_dir = Path("models") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        tracker: ExperimentTracker
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
                
                # Log metrics
                if batch_idx % self.settings.log_interval == 0:
                    step = epoch * len(train_loader) + batch_idx
                    tracker.log_metrics({
                        'train_loss': loss.item(),
                        'train_acc': 100. * correct / total
                    }, step=step)
        
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_acc': 100. * correct / total
        }
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        tracker: ExperimentTracker
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        metrics = {
            'val_loss': val_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }
        
        # Log validation metrics
        tracker.log_metrics(metrics, step=epoch)
        
        return metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        run_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, float], Path]:
        """Train the model."""
        run_params = run_params or {}
        best_metrics = None
        
        with ExperimentTracker(self.experiment_name) as tracker:
            tracker.start_run(run_params)
            
            for epoch in range(num_epochs):
                # Train and validate
                train_metrics = self.train_epoch(train_loader, epoch, tracker)
                val_metrics = self.validate(val_loader, epoch, tracker)
                
                # Check if best model
                current_metric = val_metrics['val_loss']
                is_best = current_metric < self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    best_metrics = val_metrics
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Log metrics
                tracker.log_metrics({**train_metrics, **val_metrics}, step=epoch)
            
            # Save final model
            if self.best_model_path:
                tracker.log_artifact(str(self.best_model_path))
        
        return best_metrics, self.best_model_path
