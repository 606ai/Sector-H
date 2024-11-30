import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional
import torch
from datetime import datetime
import logging
from pathlib import Path
import json
import wandb
from ..config import get_settings

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Handles experiment tracking with MLflow and Weights & Biases."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        self.settings = get_settings()
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._setup_tracking()
    
    def _setup_tracking(self):
        """Initialize MLflow and W&B tracking."""
        # MLflow setup
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # W&B setup
        if self.settings.wandb_api_key:
            wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                config=self.settings.dict(),
                sync_tensorboard=True
            )
    
    def start_run(self, run_params: Dict[str, Any]):
        """Start a new tracking run."""
        mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(run_params)
        
        # Save run parameters
        params_path = Path("params.json")
        with params_path.open("w") as f:
            json.dump(run_params, f, indent=2)
        mlflow.log_artifact(params_path)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to both MLflow and W&B."""
        mlflow.log_metrics(metrics, step=step)
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module, artifact_path: str):
        """Log PyTorch model."""
        mlflow.pytorch.log_model(model, artifact_path)
        if wandb.run is not None:
            torch.save(model.state_dict(), "model.pth")
            wandb.save("model.pth")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log additional artifacts."""
        mlflow.log_artifact(local_path, artifact_path)
        if wandb.run is not None:
            wandb.save(local_path)
    
    def end_run(self):
        """End the current tracking run."""
        mlflow.end_run()
        if wandb.run is not None:
            wandb.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
        if exc_type is not None:
            logger.error(f"Run failed: {exc_val}")
            return False
        return True
