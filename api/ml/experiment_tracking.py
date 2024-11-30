import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List, Union
import torch
import logging
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import wandb
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class ExperimentManager:
    """Manages experiment tracking with MLflow and Weights & Biases."""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        wandb_project: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.model_registry = ModelRegistry()
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # Set up W&B
        self.wandb_project = wandb_project
        if wandb_project:
            wandb.init(project=wandb_project, name=experiment_name)
        
        # Initialize MLflow experiment
        self.experiment = mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        run = mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=tags
        )
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow and W&B."""
        mlflow.log_params(params)
        if self.wandb_project:
            wandb.config.update(params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics to MLflow and W&B."""
        mlflow.log_metrics(metrics, step=step)
        if self.wandb_project:
            wandb.log(metrics, step=step)
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Log PyTorch model to MLflow and W&B."""
        # Log to MLflow
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )
        
        if custom_metrics:
            mlflow.log_metrics(custom_metrics)
        
        # Log to W&B
        if self.wandb_project:
            artifact = wandb.Artifact(
                name=registered_model_name or artifact_path,
                type="model"
            )
            wandb.log_artifact(artifact)
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        version: Optional[str] = None,
        stage: str = "None",
        description: Optional[str] = None
    ) -> str:
        """Register model in MLflow Model Registry."""
        try:
            # Register model
            result = mlflow.register_model(
                model_uri,
                name,
                tags={"version": version} if version else None
            )
            
            # Set stage and description
            if stage != "None":
                self.client.transition_model_version_stage(
                    name=name,
                    version=result.version,
                    stage=stage
                )
            
            if description:
                self.client.update_model_version(
                    name=name,
                    version=result.version,
                    description=description
                )
            
            # Add to local model registry
            self.model_registry.add_model(
                name=name,
                version=version or result.version,
                uri=model_uri,
                stage=stage,
                metadata={
                    "mlflow_version": result.version,
                    "description": description
                }
            )
            
            return result.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> torch.nn.Module:
        """Load model from MLflow Model Registry."""
        try:
            if version:
                model_uri = f"models:/{name}/{version}"
            elif stage:
                model_uri = f"models:/{name}/{stage}"
            else:
                model_uri = f"models:/{name}/latest"
            
            return mlflow.pytorch.load_model(model_uri)
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        metric_keys: List[str]
    ) -> pd.DataFrame:
        """Compare metrics between two runs."""
        try:
            baseline_metrics = self.client.get_run(baseline_run_id).data.metrics
            candidate_metrics = self.client.get_run(candidate_run_id).data.metrics
            
            comparison = pd.DataFrame({
                'Metric': metric_keys,
                'Baseline': [baseline_metrics.get(k, None) for k in metric_keys],
                'Candidate': [candidate_metrics.get(k, None) for k in metric_keys]
            })
            
            comparison['Difference'] = comparison['Candidate'] - comparison['Baseline']
            comparison['Improvement'] = (
                comparison['Difference'] / comparison['Baseline'] * 100
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {str(e)}")
            raise
    
    def get_best_run(
        self,
        metric_name: str,
        ascending: bool = False
    ) -> mlflow.entities.Run:
        """Get the best run based on a metric."""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            return runs[0] if runs else None
            
        except Exception as e:
            logger.error(f"Failed to get best run: {str(e)}")
            raise
    
    def log_artifacts(
        self,
        artifacts: Dict[str, Union[str, Path, bytes]],
        artifact_path: Optional[str] = None
    ):
        """Log artifacts to MLflow and W&B."""
        try:
            # Create temporary directory for artifacts
            tmp_dir = Path("tmp_artifacts")
            tmp_dir.mkdir(exist_ok=True)
            
            for name, content in artifacts.items():
                artifact_file = tmp_dir / name
                
                if isinstance(content, (str, Path)):
                    # File path
                    if Path(content).exists():
                        mlflow.log_artifact(content, artifact_path)
                        if self.wandb_project:
                            wandb.save(str(content))
                else:
                    # Content
                    with artifact_file.open("wb") as f:
                        if isinstance(content, str):
                            f.write(content.encode())
                        else:
                            f.write(content)
                    
                    mlflow.log_artifact(artifact_file, artifact_path)
                    if self.wandb_project:
                        wandb.save(str(artifact_file))
            
            # Clean up
            for file in tmp_dir.iterdir():
                file.unlink()
            tmp_dir.rmdir()
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {str(e)}")
            raise
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        if self.wandb_project:
            wandb.finish()
