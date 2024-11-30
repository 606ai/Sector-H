import torch
from typing import Dict, Any, Optional, Union, List
import numpy as np
from pathlib import Path
import time
from prometheus_client import Histogram, Gauge
import logging
from ..config import get_settings

logger = logging.getLogger(__name__)

# Prometheus metrics
INFERENCE_TIME = Histogram(
    'model_inference_time_seconds',
    'Time spent performing inference'
)
MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy'
)

class ModelInference:
    """Handles model inference with monitoring."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[str] = None,
        model_path: Optional[Union[str, Path]] = None
    ):
        self.settings = get_settings()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: Union[str, Path]):
        """Load model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'metrics' in checkpoint and 'val_acc' in checkpoint['metrics']:
                MODEL_ACCURACY.set(checkpoint['metrics']['val_acc'] / 100.0)
        else:
            self.model.load_state_dict(checkpoint)
    
    @torch.no_grad()
    def predict(
        self,
        input_data: torch.Tensor,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform model inference."""
        start_time = time.time()
        
        try:
            # Move input to device
            input_data = input_data.to(self.device)
            
            # Perform inference
            output = self.model(input_data)
            
            # Calculate predictions and probabilities
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            # Record inference time
            inference_time = time.time() - start_time
            INFERENCE_TIME.observe(inference_time)
            
            if return_probs:
                return preds, probs
            return preds
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict_batch(
        self,
        batch_data: List[torch.Tensor],
        batch_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Perform batch inference."""
        batch_size = batch_size or self.settings.inference_batch_size
        predictions = []
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            batch_tensor = torch.stack(batch)
            batch_preds = self.predict(batch_tensor)
            predictions.extend(batch_preds.cpu().split(1))
        
        return predictions
    
    def warmup(self, input_shape: tuple, num_warmup: int = 5):
        """Perform model warmup."""
        logger.info(f"Warming up model with {num_warmup} inferences...")
        dummy_input = torch.randn(input_shape).to(self.device)
        
        for _ in range(num_warmup):
            self.predict(dummy_input)
        
        logger.info("Model warmup complete")
