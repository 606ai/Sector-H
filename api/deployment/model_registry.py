from typing import Dict, Optional, List, Any
import mlflow
from mlflow.tracking import MlflowClient
import torch
import json
from pathlib import Path
import logging
from datetime import datetime
import boto3
from ..config import get_settings

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manages model versioning and deployment."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = MlflowClient()
        self.s3 = boto3.client('s3')
        self._setup_registry()
    
    def _setup_registry(self):
        """Initialize model registry."""
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        self.registry_path = Path("model_registry")
        self.registry_path.mkdir(parents=True, exist_ok=True)
    
    def register_model(
        self,
        model_path: Path,
        name: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Register a new model version."""
        try:
            # Log model to MLflow
            artifact_path = f"models/{name}"
            mlflow.pytorch.log_model(
                torch.load(model_path),
                artifact_path,
                registered_model_name=name
            )
            
            # Get latest version
            versions = self.client.search_model_versions(f"name='{name}'")
            version = max(int(v.version) for v in versions)
            
            # Save metadata
            metadata_path = self.registry_path / f"{name}_v{version}_metadata.json"
            metadata.update({
                "timestamp": datetime.now().isoformat(),
                "version": version
            })
            with metadata_path.open("w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Registered model {name} version {version}")
            return f"{name}_v{version}"
        
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def promote_model(
        self,
        name: str,
        version: str,
        stage: str = "Production"
    ):
        """Promote model to production/staging."""
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )
            logger.info(f"Promoted {name} version {version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {str(e)}")
            raise
    
    def deploy_model(
        self,
        name: str,
        version: str,
        deployment_config: Optional[Dict[str, Any]] = None
    ):
        """Deploy model to production environment."""
        try:
            # Get model URI
            model_uri = f"models:/{name}/{version}"
            
            # Upload to S3
            bucket = self.settings.model_bucket
            key = f"deployments/{name}/{version}/model.pth"
            
            mlflow.pytorch.save_model(
                mlflow.pytorch.load_model(model_uri),
                self.registry_path / "temp_deployment"
            )
            
            self.s3.upload_file(
                str(self.registry_path / "temp_deployment" / "model.pth"),
                bucket,
                key
            )
            
            # Update deployment config
            config = deployment_config or {}
            config.update({
                "model_uri": f"s3://{bucket}/{key}",
                "timestamp": datetime.now().isoformat(),
                "version": version
            })
            
            config_key = f"deployments/{name}/{version}/config.json"
            self.s3.put_object(
                Bucket=bucket,
                Key=config_key,
                Body=json.dumps(config)
            )
            
            logger.info(f"Deployed {name} version {version}")
            return config
        
        except Exception as e:
            logger.error(f"Failed to deploy model: {str(e)}")
            raise
    
    def rollback_deployment(self, name: str, target_version: str):
        """Rollback to a previous model version."""
        try:
            # Get current production version
            versions = self.client.search_model_versions(
                f"name='{name}' AND stage='Production'"
            )
            if versions:
                current_version = versions[0].version
                # Transition current version to Archived
                self.client.transition_model_version_stage(
                    name=name,
                    version=current_version,
                    stage="Archived"
                )
            
            # Promote target version to Production
            self.promote_model(name, target_version, "Production")
            self.deploy_model(name, target_version)
            
            logger.info(f"Rolled back {name} to version {target_version}")
        except Exception as e:
            logger.error(f"Failed to rollback: {str(e)}")
            raise
    
    def get_model_info(self, name: str, version: Optional[str] = None) -> Dict:
        """Get model information and metrics."""
        try:
            if version:
                versions = [v for v in self.client.search_model_versions(f"name='{name}'")
                          if v.version == version]
            else:
                versions = self.client.search_model_versions(f"name='{name}'")
            
            if not versions:
                raise ValueError(f"No versions found for model {name}")
            
            return {
                "name": name,
                "versions": [{
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "timestamp": v.creation_timestamp,
                } for v in versions]
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise
