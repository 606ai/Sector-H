from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from dataclasses import dataclass
from copy import deepcopy
from ..experiment_tracking import ExperimentManager

logger = logging.getLogger(__name__)

@dataclass
class TaskData:
    """Represents data for a single task."""
    x: torch.Tensor
    y: torch.Tensor
    task_id: Optional[str] = None

class Reptile(nn.Module):
    """Reptile meta-learning algorithm implementation."""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        experiment_name: Optional[str] = None
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        self.experiment_manager = ExperimentManager(
            experiment_name=experiment_name or "reptile"
        )
        
        self.metrics_history: List[Dict[str, float]] = []
    
    def inner_loop_update(
        self,
        task_data: TaskData,
        num_steps: Optional[int] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Perform inner loop optimization on a task."""
        num_steps = num_steps or self.num_inner_steps
        
        # Create task-specific model copy
        task_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(
            task_model.parameters(),
            lr=self.inner_lr
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        task_model.train()
        losses = []
        accuracies = []
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            pred = task_model(task_data.x)
            loss = criterion(pred, task_data.y)
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                pred = pred.argmax(dim=1)
                accuracy = pred.eq(task_data.y).float().mean()
            
            losses.append(loss.item())
            accuracies.append(accuracy.item())
        
        metrics = {
            "inner_loss": np.mean(losses),
            "inner_accuracy": np.mean(accuracies)
        }
        
        return task_model, metrics
    
    def meta_update(
        self,
        task_model: nn.Module,
        meta_step_size: Optional[float] = None
    ):
        """Perform meta-update using Reptile algorithm."""
        meta_step_size = meta_step_size or self.meta_lr
        
        # Compute parameter updates
        for p_meta, p_task in zip(
            self.model.parameters(),
            task_model.parameters()
        ):
            if p_meta.grad is None:
                p_meta.grad = torch.zeros_like(p_meta)
            
            # Reptile update rule
            p_meta.grad.data.add_(
                (p_task - p_meta) / meta_step_size
            )
    
    def train_step(
        self,
        task_batch: List[TaskData],
        meta_step_size: Optional[float] = None
    ) -> Dict[str, float]:
        """Perform single meta-training step."""
        batch_metrics = []
        
        # Initialize meta-gradients
        for p in self.model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            else:
                p.grad.zero_()
        
        # Process each task
        for task_data in task_batch:
            # Inner loop optimization
            task_model, metrics = self.inner_loop_update(task_data)
            batch_metrics.append(metrics)
            
            # Accumulate meta-gradients
            self.meta_update(task_model, meta_step_size)
        
        # Average meta-gradients
        for p in self.model.parameters():
            p.grad.div_(len(task_batch))
        
        # Meta-update
        for p in self.model.parameters():
            p.data.add_(p.grad, alpha=-meta_step_size or -self.meta_lr)
        
        # Compute average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in batch_metrics])
            for k in batch_metrics[0].keys()
        }
        
        return avg_metrics
    
    def evaluate(
        self,
        task_data: TaskData,
        num_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on a task."""
        self.model.eval()
        with torch.no_grad():
            # Adapt to task
            adapted_model, _ = self.inner_loop_update(
                task_data,
                num_steps
            )
            
            # Evaluate
            criterion = nn.CrossEntropyLoss()
            pred = adapted_model(task_data.x)
            loss = criterion(pred, task_data.y)
            
            # Compute accuracy
            pred = pred.argmax(dim=1)
            accuracy = pred.eq(task_data.y).float().mean()
            
            metrics = {
                "eval_loss": loss.item(),
                "eval_accuracy": accuracy.item()
            }
            
            return metrics
    
    def fit(
        self,
        task_generator: Callable[[], List[TaskData]],
        num_epochs: int,
        tasks_per_epoch: int,
        val_task_generator: Optional[Callable[[], List[TaskData]]] = None,
        val_tasks: int = 50,
        meta_step_size: Optional[float] = None
    ):
        """Train the meta-learning model."""
        try:
            for epoch in range(num_epochs):
                self.model.train()
                epoch_metrics = []
                
                # Training loop
                for _ in range(tasks_per_epoch):
                    # Generate task batch
                    task_batch = task_generator()
                    
                    # Meta-training step
                    metrics = self.train_step(
                        task_batch,
                        meta_step_size
                    )
                    epoch_metrics.append(metrics)
                
                # Compute average metrics
                avg_metrics = {
                    k: np.mean([m[k] for m in epoch_metrics])
                    for k in epoch_metrics[0].keys()
                }
                
                # Validation
                if val_task_generator is not None:
                    val_metrics = []
                    for _ in range(val_tasks):
                        task_batch = val_task_generator()
                        for task_data in task_batch:
                            metrics = self.evaluate(task_data)
                            val_metrics.append(metrics)
                    
                    # Add validation metrics
                    avg_metrics.update({
                        f"val_{k}": np.mean([m[k] for m in val_metrics])
                        for k in val_metrics[0].keys()
                    })
                
                # Log metrics
                self.experiment_manager.log_metrics(
                    avg_metrics,
                    step=epoch
                )
                
                self.metrics_history.append(avg_metrics)
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}")
                
                # Log progress
                log_str = f"Epoch {epoch + 1}/{num_epochs} - "
                log_str += " - ".join(
                    f"{k}: {v:.4f}"
                    for k, v in avg_metrics.items()
                )
                logger.info(log_str)
                
        except Exception as e:
            logger.error(f"Meta-training failed: {str(e)}")
            raise
    
    def adapt_to_task(
        self,
        task_data: TaskData,
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt the meta-learned model to a new task."""
        adapted_model, _ = self.inner_loop_update(
            task_data,
            num_steps
        )
        return adapted_model
    
    def save_checkpoint(self, tag: str):
        """Save meta-learning checkpoint."""
        try:
            checkpoint = {
                "model_state": self.model.state_dict(),
                "inner_lr": self.inner_lr,
                "meta_lr": self.meta_lr,
                "num_inner_steps": self.num_inner_steps,
                "metrics_history": self.metrics_history
            }
            
            self.experiment_manager.log_model(
                self.model,
                f"reptile_{tag}",
                custom_metrics={"tag": tag}
            )
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load meta-learning checkpoint."""
        try:
            self.model.load_state_dict(checkpoint["model_state"])
            self.inner_lr = checkpoint["inner_lr"]
            self.meta_lr = checkpoint["meta_lr"]
            self.num_inner_steps = checkpoint["num_inner_steps"]
            self.metrics_history = checkpoint["metrics_history"]
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
