from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from dataclasses import dataclass
from ..experiment_tracking import ExperimentManager

logger = logging.getLogger(__name__)

@dataclass
class EpisodeBatch:
    """Represents a batch of few-shot episodes."""
    support_x: torch.Tensor  # Shape: [batch_size, n_way, n_shot, *dim]
    support_y: torch.Tensor  # Shape: [batch_size, n_way, n_shot]
    query_x: torch.Tensor    # Shape: [batch_size, n_way, n_query, *dim]
    query_y: torch.Tensor    # Shape: [batch_size, n_way, n_query]

class ProtoNet(nn.Module):
    """Prototypical Networks for Few-shot Learning."""
    
    def __init__(
        self,
        encoder: nn.Module,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        learning_rate: float = 0.001,
        distance_metric: str = 'euclidean',
        experiment_name: Optional[str] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.learning_rate = learning_rate
        self.distance_metric = distance_metric
        
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=learning_rate
        )
        
        self.experiment_manager = ExperimentManager(
            experiment_name=experiment_name or "prototypical_networks"
        )
        
        self.metrics_history: List[Dict[str, float]] = []
    
    def compute_prototypes(
        self,
        support_x: torch.Tensor,
        n_way: int,
        n_shot: int
    ) -> torch.Tensor:
        """Compute class prototypes from support set."""
        # Encode support set
        z_support = self.encoder(
            support_x.view(-1, *support_x.shape[3:])
        )
        z_support = z_support.view(n_way, n_shot, -1)
        
        # Compute prototypes
        z_proto = z_support.mean(dim=1)
        return z_proto
    
    def compute_distances(
        self,
        z_query: torch.Tensor,
        z_proto: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances between query points and prototypes."""
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            dists = torch.cdist(z_query, z_proto)
            return -dists  # Negative distance for compatibility with softmax
            
        elif self.distance_metric == 'cosine':
            # Cosine similarity
            z_query_norm = F.normalize(z_query, dim=1)
            z_proto_norm = F.normalize(z_proto, dim=1)
            return torch.mm(z_query_norm, z_proto_norm.t())
            
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(
        self,
        episode_batch: EpisodeBatch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process a batch of few-shot episodes."""
        batch_size = episode_batch.support_x.shape[0]
        total_loss = 0
        total_acc = 0
        
        for i in range(batch_size):
            support_x = episode_batch.support_x[i]
            query_x = episode_batch.query_x[i]
            query_y = episode_batch.query_y[i]
            
            # Compute prototypes
            z_proto = self.compute_prototypes(
                support_x,
                self.n_way,
                self.n_shot
            )
            
            # Encode query set
            z_query = self.encoder(
                query_x.view(-1, *query_x.shape[2:])
            )
            
            # Compute logits
            logits = self.compute_distances(z_query, z_proto)
            
            # Compute loss
            query_y_flat = query_y.view(-1)
            loss = F.cross_entropy(logits, query_y_flat)
            total_loss += loss
            
            # Compute accuracy
            pred = logits.argmax(dim=1)
            acc = pred.eq(query_y_flat).float().mean()
            total_acc += acc
        
        # Average over batch
        avg_loss = total_loss / batch_size
        avg_acc = total_acc / batch_size
        
        metrics = {
            "loss": avg_loss.item(),
            "accuracy": avg_acc.item()
        }
        
        return avg_loss, metrics
    
    def train_step(
        self,
        episode_batch: EpisodeBatch
    ) -> Dict[str, float]:
        """Perform single training step."""
        self.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self.forward(episode_batch)
        loss.backward()
        self.optimizer.step()
        
        return metrics
    
    def validate_step(
        self,
        episode_batch: EpisodeBatch
    ) -> Dict[str, float]:
        """Perform single validation step."""
        self.eval()
        with torch.no_grad():
            _, metrics = self.forward(episode_batch)
        return metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        train_metrics = []
        
        # Training
        for episode_batch in train_loader:
            metrics = self.train_step(episode_batch)
            train_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in train_metrics])
            for k in train_metrics[0].keys()
        }
        
        # Validation
        if val_loader is not None:
            val_metrics = []
            for episode_batch in val_loader:
                metrics = self.validate_step(episode_batch)
                val_metrics.append(metrics)
            
            # Add validation metrics
            avg_metrics.update({
                f"val_{k}": np.mean([m[k] for m in val_metrics])
                for k in val_metrics[0].keys()
            })
        
        return avg_metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        save_freq: int = 10
    ):
        """Train the prototypical network."""
        try:
            for epoch in range(num_epochs):
                # Train epoch
                metrics = self.train_epoch(train_loader, val_loader)
                self.metrics_history.append(metrics)
                
                # Log metrics
                self.experiment_manager.log_metrics(
                    metrics,
                    step=epoch
                )
                
                # Save checkpoint
                if (epoch + 1) % save_freq == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}")
                
                # Log progress
                log_str = f"Epoch {epoch + 1}/{num_epochs} - "
                log_str += " - ".join(
                    f"{k}: {v:.4f}"
                    for k, v in metrics.items()
                )
                logger.info(log_str)
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(
        self,
        support_x: torch.Tensor,
        query_x: torch.Tensor,
        n_way: Optional[int] = None,
        n_shot: Optional[int] = None
    ) -> torch.Tensor:
        """Make predictions for query points."""
        self.eval()
        n_way = n_way or self.n_way
        n_shot = n_shot or self.n_shot
        
        with torch.no_grad():
            # Compute prototypes
            z_proto = self.compute_prototypes(
                support_x,
                n_way,
                n_shot
            )
            
            # Encode query set
            z_query = self.encoder(query_x)
            
            # Compute logits and predictions
            logits = self.compute_distances(z_query, z_proto)
            pred = logits.argmax(dim=1)
            
            return pred
    
    def save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        try:
            checkpoint = {
                "encoder_state": self.encoder.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "n_way": self.n_way,
                "n_shot": self.n_shot,
                "n_query": self.n_query,
                "learning_rate": self.learning_rate,
                "distance_metric": self.distance_metric,
                "metrics_history": self.metrics_history
            }
            
            self.experiment_manager.log_model(
                self.encoder,
                f"protonet_{tag}",
                custom_metrics={"tag": tag}
            )
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load model checkpoint."""
        try:
            self.encoder.load_state_dict(checkpoint["encoder_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.n_way = checkpoint["n_way"]
            self.n_shot = checkpoint["n_shot"]
            self.n_query = checkpoint["n_query"]
            self.learning_rate = checkpoint["learning_rate"]
            self.distance_metric = checkpoint["distance_metric"]
            self.metrics_history = checkpoint["metrics_history"]
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
