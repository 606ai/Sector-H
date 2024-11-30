from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import asyncio
import logging
import json
from dataclasses import dataclass
from datetime import datetime
import cryptography.fernet
from ..experiment_tracking import ExperimentManager

logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """Configuration for federated learning clients."""
    client_id: str
    min_samples: int
    batch_size: int
    learning_rate: float
    local_epochs: int
    dp_epsilon: Optional[float] = None
    dp_delta: Optional[float] = None
    dp_noise_multiplier: Optional[float] = None

class FederatedServer:
    """Manages federated learning across distributed clients."""
    
    def __init__(
        self,
        model: nn.Module,
        aggregation_strategy: str = "fedavg",
        min_clients: int = 3,
        rounds: int = 10,
        experiment_name: Optional[str] = None
    ):
        self.model = model
        self.aggregation_strategy = aggregation_strategy
        self.min_clients = min_clients
        self.rounds = rounds
        
        # Initialize encryption for secure aggregation
        self.key = cryptography.fernet.Fernet.generate_key()
        self.cipher = cryptography.fernet.Fernet(self.key)
        
        # Setup experiment tracking
        self.experiment_manager = ExperimentManager(
            experiment_name=experiment_name or "federated_training"
        )
        
        self.clients: Dict[str, ClientConfig] = {}
        self.current_round = 0
        self.model_updates: Dict[str, Dict[str, torch.Tensor]] = {}
        self.metrics_history: List[Dict[str, Any]] = []
    
    async def register_client(
        self,
        client_id: str,
        config: ClientConfig
    ) -> Dict[str, Any]:
        """Register a new client for federated learning."""
        try:
            self.clients[client_id] = config
            
            # Generate client-specific encryption key
            client_key = cryptography.fernet.Fernet.generate_key()
            
            return {
                "status": "success",
                "client_id": client_id,
                "encryption_key": client_key.decode(),
                "model_config": {
                    name: param.shape
                    for name, param in self.model.state_dict().items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _apply_differential_privacy(
        self,
        updates: Dict[str, torch.Tensor],
        epsilon: float,
        delta: float,
        noise_multiplier: float
    ) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to model updates."""
        try:
            dp_updates = {}
            
            for name, param in updates.items():
                # Calculate sensitivity
                sensitivity = torch.norm(param).item()
                
                # Add Gaussian noise
                noise = torch.randn_like(param) * sensitivity * noise_multiplier
                dp_updates[name] = param + noise
                
                # Clip gradients
                max_norm = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                dp_updates[name] = torch.clamp(
                    dp_updates[name],
                    -max_norm,
                    max_norm
                )
            
            return dp_updates
            
        except Exception as e:
            logger.error(f"Failed to apply differential privacy: {str(e)}")
            raise
    
    async def receive_update(
        self,
        client_id: str,
        encrypted_updates: str,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Receive and process model updates from a client."""
        try:
            if client_id not in self.clients:
                raise ValueError(f"Unknown client: {client_id}")
            
            # Decrypt updates
            updates_bytes = self.cipher.decrypt(encrypted_updates.encode())
            updates = {
                name: torch.tensor(param)
                for name, param in json.loads(updates_bytes).items()
            }
            
            # Apply differential privacy if configured
            client_config = self.clients[client_id]
            if client_config.dp_epsilon is not None:
                updates = self._apply_differential_privacy(
                    updates,
                    client_config.dp_epsilon,
                    client_config.dp_delta,
                    client_config.dp_noise_multiplier
                )
            
            # Store updates
            self.model_updates[client_id] = updates
            
            # Log metrics
            self.experiment_manager.log_metrics({
                f"{client_id}_{k}": v
                for k, v in metrics.items()
            })
            
            # Check if ready for aggregation
            if len(self.model_updates) >= self.min_clients:
                await self.aggregate_updates()
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Failed to process update from {client_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def aggregate_updates(self):
        """Aggregate model updates from clients."""
        try:
            if self.aggregation_strategy == "fedavg":
                # FedAvg aggregation
                aggregated_updates = {}
                
                for name, param in self.model.state_dict().items():
                    updates = [
                        client_updates[name]
                        for client_updates in self.model_updates.values()
                    ]
                    aggregated_updates[name] = torch.stack(updates).mean(dim=0)
                
                # Update global model
                self.model.load_state_dict(aggregated_updates)
            
            elif self.aggregation_strategy == "fedprox":
                # FedProx aggregation with proximal term
                mu = 0.01  # proximal term weight
                aggregated_updates = {}
                
                for name, param in self.model.state_dict().items():
                    updates = [
                        client_updates[name]
                        for client_updates in self.model_updates.values()
                    ]
                    
                    # Add proximal term
                    prox_term = mu * (torch.stack(updates) - param)
                    aggregated_updates[name] = (
                        torch.stack(updates).mean(dim=0) + prox_term.mean(dim=0)
                    )
                
                self.model.load_state_dict(aggregated_updates)
            
            # Clear updates
            self.model_updates.clear()
            self.current_round += 1
            
            # Save checkpoint
            self.save_checkpoint()
            
        except Exception as e:
            logger.error(f"Failed to aggregate updates: {str(e)}")
            raise
    
    def save_checkpoint(self):
        """Save model checkpoint and training state."""
        try:
            checkpoint = {
                "model_state": self.model.state_dict(),
                "current_round": self.current_round,
                "clients": self.clients,
                "metrics_history": self.metrics_history,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to MLflow
            with self.experiment_manager.start_run(
                run_name=f"round_{self.current_round}"
            ):
                self.experiment_manager.log_model(
                    self.model,
                    f"model_round_{self.current_round}",
                    custom_metrics={
                        "round": self.current_round,
                        "n_clients": len(self.clients)
                    }
                )
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise
    
    async def get_model_update(
        self,
        client_id: str
    ) -> Dict[str, Any]:
        """Get latest model update for a client."""
        try:
            if client_id not in self.clients:
                raise ValueError(f"Unknown client: {client_id}")
            
            # Encrypt model state
            model_state = {
                name: param.cpu().numpy().tolist()
                for name, param in self.model.state_dict().items()
            }
            
            encrypted_state = self.cipher.encrypt(
                json.dumps(model_state).encode()
            )
            
            return {
                "status": "success",
                "model_update": encrypted_state.decode(),
                "round": self.current_round
            }
            
        except Exception as e:
            logger.error(f"Failed to get model update for {client_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress and metrics."""
        try:
            return {
                "current_round": self.current_round,
                "total_rounds": self.rounds,
                "n_clients": len(self.clients),
                "metrics_history": self.metrics_history,
                "aggregation_strategy": self.aggregation_strategy
            }
            
        except Exception as e:
            logger.error(f"Failed to get training progress: {str(e)}")
            return {"status": "error", "message": str(e)}
