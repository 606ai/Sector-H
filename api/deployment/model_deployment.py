from typing import Dict, Any, Optional, List, Union
import torch
import logging
from pathlib import Path
import json
import docker
import kubernetes as k8s
from kubernetes import client, config
from prometheus_client import Counter, Histogram
import mlflow
from ..ml.experiment_tracking import ExperimentManager
from .ab_testing import ABTestingManager

logger = logging.getLogger(__name__)

# Prometheus metrics
DEPLOYMENT_COUNT = Counter(
    'model_deployments_total',
    'Total number of model deployments',
    ['model_name', 'version', 'environment']
)
DEPLOYMENT_LATENCY = Histogram(
    'model_deployment_latency_seconds',
    'Model deployment latency',
    ['model_name', 'version', 'environment']
)

class ModelDeployment:
    """Handles model deployment to various environments."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        docker_registry: Optional[str] = None
    ):
        self.config = self._load_config(config_path)
        self.docker_client = docker.from_env()
        self.docker_registry = docker_registry
        self.experiment_manager = ExperimentManager(
            experiment_name="model_deployments"
        )
        self.ab_testing = ABTestingManager()
        
        # Initialize Kubernetes client if configured
        if self.config.get("use_kubernetes", False):
            try:
                config.load_kube_config()
                self.k8s_client = client.ApiClient()
                self.k8s_apps = client.AppsV1Api()
                self.k8s_core = client.CoreV1Api()
            except Exception as e:
                logger.warning(f"Failed to initialize Kubernetes client: {str(e)}")
                self.k8s_client = None
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict:
        """Load deployment configuration."""
        if config_path:
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def build_docker_image(
        self,
        model_name: str,
        version: str,
        model_uri: str,
        requirements: Optional[List[str]] = None,
        cuda_version: Optional[str] = None
    ) -> str:
        """Build Docker image for model deployment."""
        try:
            # Create temporary directory for build
            build_dir = Path("docker_build")
            build_dir.mkdir(exist_ok=True)
            
            # Create Dockerfile
            dockerfile = [
                f"FROM {'nvidia/cuda:' + cuda_version if cuda_version else 'python:3.10-slim'}",
                "WORKDIR /app",
                "COPY requirements.txt .",
                "RUN pip install -r requirements.txt",
                "COPY model .",
                "EXPOSE 8000",
                'CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]'
            ]
            
            with (build_dir / "Dockerfile").open("w") as f:
                f.write("\n".join(dockerfile))
            
            # Create requirements.txt
            base_requirements = [
                "fastapi",
                "uvicorn",
                "torch",
                "mlflow",
                "prometheus_client"
            ]
            if requirements:
                base_requirements.extend(requirements)
            
            with (build_dir / "requirements.txt").open("w") as f:
                f.write("\n".join(base_requirements))
            
            # Download model from MLflow
            model = mlflow.pytorch.load_model(model_uri)
            torch.save(model.state_dict(), build_dir / "model.pt")
            
            # Create FastAPI app
            app_code = """
from fastapi import FastAPI, HTTPException
import torch
import mlflow
import prometheus_client as prom
import time

app = FastAPI()

# Load model
model = mlflow.pytorch.load_model("model.pt")
model.eval()

# Metrics
INFERENCE_REQUESTS = prom.Counter(
    'model_inference_requests_total',
    'Total number of inference requests',
    ['model_name', 'version']
)
INFERENCE_LATENCY = prom.Histogram(
    'model_inference_latency_seconds',
    'Model inference latency',
    ['model_name', 'version']
)

@app.post("/predict")
async def predict(data: dict):
    try:
        start_time = time.time()
        
        # Convert input to tensor
        input_tensor = torch.tensor(data["input"])
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Update metrics
        INFERENCE_REQUESTS.labels(
            model_name="{model_name}",
            version="{version}"
        ).inc()
        
        INFERENCE_LATENCY.labels(
            model_name="{model_name}",
            version="{version}"
        ).observe(time.time() - start_time)
        
        return {"prediction": output.tolist()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return prom.generate_latest()
""".format(model_name=model_name, version=version)
            
            with (build_dir / "app.py").open("w") as f:
                f.write(app_code)
            
            # Build Docker image
            image_name = f"{model_name}:{version}"
            if self.docker_registry:
                image_name = f"{self.docker_registry}/{image_name}"
            
            self.docker_client.images.build(
                path=str(build_dir),
                tag=image_name,
                rm=True
            )
            
            # Clean up
            for file in build_dir.iterdir():
                file.unlink()
            build_dir.rmdir()
            
            return image_name
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {str(e)}")
            raise
    
    def deploy_to_kubernetes(
        self,
        image_name: str,
        model_name: str,
        version: str,
        replicas: int = 1,
        resources: Optional[Dict[str, Any]] = None,
        environment: str = "production"
    ):
        """Deploy model to Kubernetes cluster."""
        try:
            if not self.k8s_client:
                raise ValueError("Kubernetes client not initialized")
            
            # Create deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=f"{model_name}-{version}".lower(),
                    labels={
                        "app": model_name,
                        "version": version
                    }
                ),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={
                            "app": model_name,
                            "version": version
                        }
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={
                                "app": model_name,
                                "version": version
                            }
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=model_name,
                                    image=image_name,
                                    ports=[
                                        client.V1ContainerPort(
                                            container_port=8000
                                        )
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        **resources
                                    ) if resources else None
                                )
                            ]
                        )
                    )
                )
            )
            
            # Create service
            service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=f"{model_name}-{version}-svc".lower()
                ),
                spec=client.V1ServiceSpec(
                    selector={
                        "app": model_name,
                        "version": version
                    },
                    ports=[
                        client.V1ServicePort(
                            port=8000,
                            target_port=8000
                        )
                    ]
                )
            )
            
            # Deploy
            self.k8s_apps.create_namespaced_deployment(
                namespace="default",
                body=deployment
            )
            
            self.k8s_core.create_namespaced_service(
                namespace="default",
                body=service
            )
            
            # Update metrics
            DEPLOYMENT_COUNT.labels(
                model_name=model_name,
                version=version,
                environment=environment
            ).inc()
            
            # Log deployment to MLflow
            with self.experiment_manager.start_run(
                run_name=f"deploy_{model_name}_{version}"
            ):
                self.experiment_manager.log_params({
                    "model_name": model_name,
                    "version": version,
                    "environment": environment,
                    "replicas": replicas,
                    "resources": resources
                })
            
        except Exception as e:
            logger.error(f"Failed to deploy to Kubernetes: {str(e)}")
            raise
    
    def create_ab_test(
        self,
        experiment_id: str,
        model_a: Dict[str, Any],
        model_b: Dict[str, Any],
        traffic_split: Optional[Dict[str, float]] = None
    ):
        """Create A/B test between two model versions."""
        try:
            # Deploy both models
            for model in [model_a, model_b]:
                self.deploy_to_kubernetes(
                    image_name=model["image"],
                    model_name=model["name"],
                    version=model["version"],
                    replicas=model.get("replicas", 1),
                    resources=model.get("resources")
                )
            
            # Create A/B test
            self.ab_testing.create_experiment(
                experiment_id=experiment_id,
                variants={
                    "A": model_a,
                    "B": model_b
                },
                traffic_split=traffic_split
            )
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {str(e)}")
            raise
    
    def rollback(
        self,
        model_name: str,
        target_version: str
    ):
        """Rollback to a previous model version."""
        try:
            if not self.k8s_client:
                raise ValueError("Kubernetes client not initialized")
            
            # Update deployment
            deployment = self.k8s_apps.read_namespaced_deployment(
                name=f"{model_name}-{target_version}".lower(),
                namespace="default"
            )
            
            self.k8s_apps.replace_namespaced_deployment(
                name=f"{model_name}-{target_version}".lower(),
                namespace="default",
                body=deployment
            )
            
            # Log rollback to MLflow
            with self.experiment_manager.start_run(
                run_name=f"rollback_{model_name}_{target_version}"
            ):
                self.experiment_manager.log_params({
                    "model_name": model_name,
                    "target_version": target_version,
                    "action": "rollback"
                })
            
        except Exception as e:
            logger.error(f"Failed to rollback: {str(e)}")
            raise
