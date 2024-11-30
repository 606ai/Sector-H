import pytest
import torch
import mlflow
from unittest.mock import Mock, patch
from pathlib import Path
import json
import docker
from kubernetes import client
from ..deployment.model_deployment import ModelDeployment
from ..ml.experiment_tracking import ExperimentManager

@pytest.fixture
def model_deployment():
    return ModelDeployment(
        config_path=None,
        docker_registry="test-registry"
    )

@pytest.fixture
def simple_model():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    return SimpleModel()

def test_build_docker_image(model_deployment, simple_model):
    with patch('docker.from_env') as mock_docker:
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock MLflow
        with patch('mlflow.pytorch.load_model') as mock_load_model:
            mock_load_model.return_value = simple_model
            
            image_name = model_deployment.build_docker_image(
                model_name="test-model",
                version="v1",
                model_uri="models:/test-model/v1"
            )
            
            assert image_name == "test-registry/test-model:v1"
            mock_client.images.build.assert_called_once()

def test_deploy_to_kubernetes(model_deployment):
    with patch('kubernetes.client.AppsV1Api') as mock_apps_api:
        with patch('kubernetes.client.CoreV1Api') as mock_core_api:
            mock_apps = Mock()
            mock_core = Mock()
            mock_apps_api.return_value = mock_apps
            mock_core_api.return_value = mock_core
            
            model_deployment.deploy_to_kubernetes(
                image_name="test-registry/test-model:v1",
                model_name="test-model",
                version="v1",
                replicas=2
            )
            
            mock_apps.create_namespaced_deployment.assert_called_once()
            mock_core.create_namespaced_service.assert_called_once()

def test_create_ab_test(model_deployment):
    with patch('kubernetes.client.AppsV1Api') as mock_apps_api:
        with patch('kubernetes.client.CoreV1Api') as mock_core_api:
            with patch('api.deployment.ab_testing.ABTestingManager') as mock_ab:
                mock_apps = Mock()
                mock_core = Mock()
                mock_ab_instance = Mock()
                mock_apps_api.return_value = mock_apps
                mock_core_api.return_value = mock_core
                mock_ab.return_value = mock_ab_instance
                
                model_deployment.create_ab_test(
                    experiment_id="test-experiment",
                    model_a={
                        "name": "model-a",
                        "version": "v1",
                        "image": "test-registry/model-a:v1"
                    },
                    model_b={
                        "name": "model-b",
                        "version": "v1",
                        "image": "test-registry/model-b:v1"
                    }
                )
                
                assert mock_apps.create_namespaced_deployment.call_count == 2
                assert mock_core.create_namespaced_service.call_count == 2
                mock_ab_instance.create_experiment.assert_called_once()

def test_rollback(model_deployment):
    with patch('kubernetes.client.AppsV1Api') as mock_apps_api:
        mock_apps = Mock()
        mock_apps_api.return_value = mock_apps
        
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name="test-model-v1"
            )
        )
        mock_apps.read_namespaced_deployment.return_value = deployment
        
        model_deployment.rollback(
            model_name="test-model",
            target_version="v1"
        )
        
        mock_apps.read_namespaced_deployment.assert_called_once()
        mock_apps.replace_namespaced_deployment.assert_called_once()
