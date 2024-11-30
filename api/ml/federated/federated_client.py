from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import logging
import asyncio
import cryptography.fernet
from ..experiment_tracking import ExperimentManager

logger = logging.getLogger(__name__)

class FederatedClient:
    """Client for federated learning training."""
    
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        client_id: str,
        server_url: str,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
        dp_epsilon: Optional[float] = None,
        dp_delta: Optional[float] = None,
        dp_noise_multiplier: Optional[float] = None
    ):
        self.model = model
        self.dataset = dataset
        self.client_id = client_id
        self.server_url = server_url
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        
        # Differential privacy settings
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_noise_multiplier = dp_noise_multiplier
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.experiment_manager = ExperimentManager(
            experiment_name=f"federated_client_{client_id}"
        )
        
        self.encryption_key = None
        self.cipher = None
        self.current_round = 0
    
    async def register_with_server(self) -> bool:
        """Register with federated learning server."""
        try:
            config = {
                "client_id": self.client_id,
                "min_samples": len(self.dataset),
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "local_epochs": self.local_epochs,
                "dp_epsilon": self.dp_epsilon,
                "dp_delta": self.dp_delta,
                "dp_noise_multiplier": self.dp_noise_multiplier
            }
            
            # Send registration request
            response = await self._send_request(
                "register",
                config
            )
            
            if response["status"] == "success":
                self.encryption_key = response["encryption_key"].encode()
                self.cipher = cryptography.fernet.Fernet(self.encryption_key)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to register with server: {str(e)}")
            return False
    
    async def train_local(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform local training for one round."""
        try:
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Local training loop
            for epoch in range(self.local_epochs):
                for batch_idx, (data, target) in enumerate(self.dataloader):
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping for DP
                    if self.dp_epsilon is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=1.0
                        )
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            # Calculate metrics
            metrics = {
                "loss": total_loss / len(self.dataloader),
                "accuracy": correct / total
            }
            
            # Get model updates
            updates = {
                name: param.cpu()
                for name, param in self.model.state_dict().items()
            }
            
            return updates, metrics
            
        except Exception as e:
            logger.error(f"Failed local training: {str(e)}")
            raise
    
    async def send_update(
        self,
        updates: Dict[str, torch.Tensor],
        metrics: Dict[str, float]
    ) -> bool:
        """Send model updates to server."""
        try:
            # Convert tensors to lists for JSON serialization
            updates_list = {
                name: param.numpy().tolist()
                for name, param in updates.items()
            }
            
            # Encrypt updates
            encrypted_updates = self.cipher.encrypt(
                json.dumps(updates_list).encode()
            ).decode()
            
            # Send to server
            response = await self._send_request(
                "update",
                {
                    "client_id": self.client_id,
                    "encrypted_updates": encrypted_updates,
                    "metrics": metrics
                }
            )
            
            return response["status"] == "success"
            
        except Exception as e:
            logger.error(f"Failed to send update: {str(e)}")
            return False
    
    async def get_global_model(self) -> bool:
        """Get latest global model from server."""
        try:
            response = await self._send_request(
                "get_model",
                {"client_id": self.client_id}
            )
            
            if response["status"] == "success":
                # Decrypt model state
                encrypted_state = response["model_update"].encode()
                state_dict = json.loads(
                    self.cipher.decrypt(encrypted_state).decode()
                )
                
                # Update local model
                new_state = {
                    name: torch.tensor(param)
                    for name, param in state_dict.items()
                }
                
                self.model.load_state_dict(new_state)
                self.current_round = response["round"]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to get global model: {str(e)}")
            return False
    
    async def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate current model on test data."""
        try:
            self.model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            metrics = {
                "test_loss": test_loss / len(test_loader),
                "test_accuracy": correct / total
            }
            
            # Log metrics
            self.experiment_manager.log_metrics(
                metrics,
                step=self.current_round
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {str(e)}")
            raise
    
    async def _send_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send HTTP request to server."""
        try:
            # Implement your HTTP client logic here
            # This is a placeholder for actual HTTP implementation
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Failed to send request: {str(e)}")
            raise
