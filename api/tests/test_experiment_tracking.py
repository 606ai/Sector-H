import pytest
import mlflow
import torch
import wandb
from unittest.mock import Mock, patch
from ..ml.experiment_tracking import ExperimentManager

@pytest.fixture
def experiment_manager():
    return ExperimentManager(
        experiment_name="test-experiment",
        wandb_project="test-project"
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

def test_start_run(experiment_manager):
    with patch('mlflow.start_run') as mock_start_run:
        mock_run = Mock()
        mock_start_run.return_value = mock_run
        
        run = experiment_manager.start_run(
            run_name="test-run",
            tags={"tag": "value"}
        )
        
        assert run == mock_run
        mock_start_run.assert_called_once_with(
            run_name="test-run",
            nested=False,
            tags={"tag": "value"}
        )

def test_log_params(experiment_manager):
    with patch('mlflow.log_params') as mock_log_params:
        with patch('wandb.config.update') as mock_wandb_update:
            params = {"param1": 1, "param2": "value"}
            
            experiment_manager.log_params(params)
            
            mock_log_params.assert_called_once_with(params)
            mock_wandb_update.assert_called_once_with(params)

def test_log_metrics(experiment_manager):
    with patch('mlflow.log_metrics') as mock_log_metrics:
        with patch('wandb.log') as mock_wandb_log:
            metrics = {"metric1": 0.9, "metric2": 0.8}
            
            experiment_manager.log_metrics(metrics, step=1)
            
            mock_log_metrics.assert_called_once_with(metrics, step=1)
            mock_wandb_log.assert_called_once_with(metrics, step=1)

def test_log_model(experiment_manager, simple_model):
    with patch('mlflow.pytorch.log_model') as mock_log_model:
        with patch('wandb.Artifact') as mock_artifact:
            with patch('wandb.log_artifact') as mock_log_artifact:
                mock_artifact_instance = Mock()
                mock_artifact.return_value = mock_artifact_instance
                
                experiment_manager.log_model(
                    model=simple_model,
                    artifact_path="models",
                    registered_model_name="test-model",
                    custom_metrics={"accuracy": 0.95}
                )
                
                mock_log_model.assert_called_once_with(
                    simple_model,
                    "models",
                    registered_model_name="test-model"
                )
                mock_artifact.assert_called_once_with(
                    name="test-model",
                    type="model"
                )
                mock_log_artifact.assert_called_once_with(mock_artifact_instance)

def test_register_model(experiment_manager):
    with patch('mlflow.register_model') as mock_register:
        with patch('mlflow.client.MlflowClient') as mock_client:
            mock_result = Mock()
            mock_result.version = "1"
            mock_register.return_value = mock_result
            
            version = experiment_manager.register_model(
                model_uri="runs:/abc/models",
                name="test-model",
                version="v1",
                stage="Production",
                description="Test model"
            )
            
            assert version == "1"
            mock_register.assert_called_once_with(
                "runs:/abc/models",
                "test-model",
                tags={"version": "v1"}
            )

def test_load_model(experiment_manager):
    with patch('mlflow.pytorch.load_model') as mock_load:
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        model = experiment_manager.load_model(
            name="test-model",
            version="v1"
        )
        
        assert model == mock_model
        mock_load.assert_called_once_with("models:/test-model/v1")

def test_compare_runs(experiment_manager):
    with patch('mlflow.client.MlflowClient') as mock_client:
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        run1 = Mock()
        run2 = Mock()
        run1.data.metrics = {"accuracy": 0.9, "loss": 0.1}
        run2.data.metrics = {"accuracy": 0.95, "loss": 0.08}
        
        mock_client_instance.get_run.side_effect = [run1, run2]
        
        comparison = experiment_manager.compare_runs(
            baseline_run_id="run1",
            candidate_run_id="run2",
            metric_keys=["accuracy", "loss"]
        )
        
        assert len(comparison) == 2
        assert comparison["Improvement"].iloc[0] == pytest.approx(5.56, rel=1e-2)
        assert comparison["Improvement"].iloc[1] == pytest.approx(-20.0, rel=1e-2)
