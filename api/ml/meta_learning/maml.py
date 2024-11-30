from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import higher
import numpy as np
import logging
from dataclasses import dataclass
from copy import deepcopy
from ..experiment_tracking import ExperimentManager

logger = logging.getLogger(__name__)

@dataclass
class TaskBatch:
    """Represents a batch of tasks for meta-learning."""
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    task_id: Optional[str] = None

class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) implementation."""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        first_order: bool = False,
        num_inner_steps: int = 5,
        task_batch_size: int = 4,
        experiment_name: Optional[str] = None
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.first_order = first_order
        self.num_inner_steps = num_inner_steps
        self.task_batch_size = task_batch_size
        
        # Initialize meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=meta_lr
        )
        
        # Setup experiment tracking
        self.experiment_manager = ExperimentManager(
            experiment_name=experiment_name or "meta_learning"
        )
        
        self.metrics_history: List[Dict[str, float]] = []
    
    def forward(
        self,
        task_batch: TaskBatch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform meta-learning on a batch of tasks."""
        meta_loss = 0.0
        meta_metrics = {}
        
        # Initialize task-specific metrics
        task_losses = []
        task_accuracies = []
        
        # Inner loop optimization
        for task_idx in range(self.task_batch_size):
            support_x = task_batch.support_x[task_idx]
            support_y = task_batch.support_y[task_idx]
            query_x = task_batch.query_x[task_idx]
            query_y = task_batch.query_y[task_idx]
            
            # Create functional model for inner loop optimization
            with higher.innerloop_ctx(
                self.model,
                self.meta_optimizer,
                copy_initial_weights=False,
                track_higher_grads=not self.first_order
            ) as (fmodel, diffopt):
                # Inner loop training
                for _ in range(self.num_inner_steps):
                    support_pred = fmodel(support_x)
                    inner_loss = F.cross_entropy(support_pred, support_y)
                    diffopt.step(inner_loss)
                
                # Evaluate on query set
                query_pred = fmodel(query_x)
                task_loss = F.cross_entropy(query_pred, query_y)
                
                # Compute accuracy
                pred = query_pred.argmax(dim=1)
                accuracy = pred.eq(query_y).float().mean()
                
                # Accumulate metrics
                task_losses.append(task_loss)
                task_accuracies.append(accuracy)
                
                meta_loss += task_loss
        
        # Average metrics
        meta_loss = meta_loss / self.task_batch_size
        meta_metrics = {
            "meta_loss": meta_loss.item(),
            "mean_task_loss": torch.stack(task_losses).mean().item(),
            "mean_task_accuracy": torch.stack(task_accuracies).mean().item()
        }
        
        return meta_loss, meta_metrics
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt the meta-learned model to a new task."""
        if num_steps is None:
            num_steps = self.num_inner_steps
        
        # Create a copy of the model for adaptation
        adapted_model = deepcopy(self.model)
        optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )
        
        # Perform adaptation steps
        adapted_model.train()
        for _ in range(num_steps):
            optimizer.zero_grad()
            pred = adapted_model(support_x)
            loss = F.cross_entropy(pred, support_y)
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_train(
        self,
        task_generator: Callable[[], TaskBatch],
        num_epochs: int,
        tasks_per_epoch: int,
        val_task_generator: Optional[Callable[[], TaskBatch]] = None,
        val_tasks: int = 50
    ):
        """Train the meta-learning model."""
        try:
            for epoch in range(num_epochs):
                self.model.train()
                epoch_losses = []
                epoch_accuracies = []
                
                # Training loop
                for task_idx in range(tasks_per_epoch):
                    # Generate task batch
                    task_batch = task_generator()
                    
                    # Meta-optimization step
                    self.meta_optimizer.zero_grad()
                    meta_loss, metrics = self.forward(task_batch)
                    meta_loss.backward()
                    self.meta_optimizer.step()
                    
                    # Record metrics
                    epoch_losses.append(metrics["meta_loss"])
                    epoch_accuracies.append(metrics["mean_task_accuracy"])
                
                # Compute epoch metrics
                epoch_metrics = {
                    "train_loss": np.mean(epoch_losses),
                    "train_accuracy": np.mean(epoch_accuracies)
                }
                
                # Validation
                if val_task_generator is not None:
                    val_metrics = self.meta_validate(
                        val_task_generator,
                        val_tasks
                    )
                    epoch_metrics.update(val_metrics)
                
                # Log metrics
                self.experiment_manager.log_metrics(
                    epoch_metrics,
                    step=epoch
                )
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}")
                
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Loss: {epoch_metrics['train_loss']:.4f} - "
                    f"Accuracy: {epoch_metrics['train_accuracy']:.4f}"
                )
                
                self.metrics_history.append(epoch_metrics)
            
        except Exception as e:
            logger.error(f"Meta-training failed: {str(e)}")
            raise
    
    def meta_validate(
        self,
        task_generator: Callable[[], TaskBatch],
        num_tasks: int
    ) -> Dict[str, float]:
        """Validate meta-learning performance."""
        self.model.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for _ in range(num_tasks):
                task_batch = task_generator()
                _, metrics = self.forward(task_batch)
                val_losses.append(metrics["meta_loss"])
                val_accuracies.append(metrics["mean_task_accuracy"])
        
        return {
            "val_loss": np.mean(val_losses),
            "val_accuracy": np.mean(val_accuracies)
        }
    
    def save_checkpoint(self, tag: str):
        """Save meta-learning checkpoint."""
        try:
            checkpoint = {
                "model_state": self.model.state_dict(),
                "meta_optimizer_state": self.meta_optimizer.state_dict(),
                "inner_lr": self.inner_lr,
                "meta_lr": self.meta_lr,
                "num_inner_steps": self.num_inner_steps,
                "metrics_history": self.metrics_history
            }
            
            self.experiment_manager.log_model(
                self.model,
                f"maml_{tag}",
                custom_metrics={"tag": tag}
            )
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load meta-learning checkpoint."""
        try:
            self.model.load_state_dict(checkpoint["model_state"])
            self.meta_optimizer.load_state_dict(
                checkpoint["meta_optimizer_state"]
            )
            self.inner_lr = checkpoint["inner_lr"]
            self.meta_lr = checkpoint["meta_lr"]
            self.num_inner_steps = checkpoint["num_inner_steps"]
            self.metrics_history = checkpoint["metrics_history"]
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
