from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OpType(Enum):
    CONV = "conv"
    SEPARABLE_CONV = "separable_conv"
    DILATED_CONV = "dilated_conv"
    ATTENTION = "attention"
    IDENTITY = "identity"
    ZERO = "zero"

@dataclass
class SearchSpace:
    """Defines the search space for neural architecture search."""
    
    input_shape: tuple
    n_classes: int
    max_layers: int
    operations: List[OpType]
    max_filters: int
    max_kernel_size: int
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random architecture from the search space."""
        n_layers = np.random.randint(1, self.max_layers + 1)
        architecture = {
            "n_layers": n_layers,
            "layers": []
        }
        
        current_shape = self.input_shape
        
        for i in range(n_layers):
            layer = self._sample_layer(current_shape)
            architecture["layers"].append(layer)
            current_shape = self._compute_output_shape(current_shape, layer)
        
        # Add final classification layer
        architecture["layers"].append({
            "type": "classification",
            "n_classes": self.n_classes
        })
        
        return architecture
    
    def _sample_layer(self, input_shape: tuple) -> Dict[str, Any]:
        """Sample a single layer configuration."""
        op_type = np.random.choice(self.operations)
        
        if op_type == OpType.CONV:
            return {
                "type": OpType.CONV.value,
                "filters": 2 ** np.random.randint(4, int(np.log2(self.max_filters)) + 1),
                "kernel_size": np.random.randint(1, self.max_kernel_size + 1),
                "stride": np.random.choice([1, 2]),
                "activation": np.random.choice(["relu", "swish"])
            }
        elif op_type == OpType.SEPARABLE_CONV:
            return {
                "type": OpType.SEPARABLE_CONV.value,
                "filters": 2 ** np.random.randint(4, int(np.log2(self.max_filters)) + 1),
                "kernel_size": np.random.randint(1, self.max_kernel_size + 1),
                "stride": np.random.choice([1, 2]),
                "activation": np.random.choice(["relu", "swish"])
            }
        elif op_type == OpType.DILATED_CONV:
            return {
                "type": OpType.DILATED_CONV.value,
                "filters": 2 ** np.random.randint(4, int(np.log2(self.max_filters)) + 1),
                "kernel_size": np.random.randint(1, self.max_kernel_size + 1),
                "dilation_rate": np.random.randint(1, 4),
                "activation": np.random.choice(["relu", "swish"])
            }
        elif op_type == OpType.ATTENTION:
            return {
                "type": OpType.ATTENTION.value,
                "heads": 2 ** np.random.randint(1, 4),
                "key_dim": 2 ** np.random.randint(4, 7)
            }
        else:
            return {
                "type": op_type.value
            }
    
    def _compute_output_shape(
        self,
        input_shape: tuple,
        layer_config: Dict[str, Any]
    ) -> tuple:
        """Compute output shape for a layer."""
        if layer_config["type"] in [OpType.CONV.value, OpType.SEPARABLE_CONV.value]:
            h, w, _ = input_shape
            stride = layer_config["stride"]
            kernel = layer_config["kernel_size"]
            padding = kernel // 2
            
            h = (h + 2 * padding - kernel) // stride + 1
            w = (w + 2 * padding - kernel) // stride + 1
            c = layer_config["filters"]
            
            return (h, w, c)
        
        elif layer_config["type"] == OpType.DILATED_CONV.value:
            h, w, _ = input_shape
            kernel = layer_config["kernel_size"]
            dilation = layer_config["dilation_rate"]
            effective_kernel = kernel + (kernel - 1) * (dilation - 1)
            padding = effective_kernel // 2
            
            h = h + 2 * padding - effective_kernel + 1
            w = w + 2 * padding - effective_kernel + 1
            c = layer_config["filters"]
            
            return (h, w, c)
        
        elif layer_config["type"] == OpType.ATTENTION.value:
            return input_shape
        
        else:
            return input_shape

class ArchitectureGenerator:
    """Generates neural network architectures based on search space."""
    
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
    
    def generate_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """Generate PyTorch model from architecture specification."""
        class DynamicModel(nn.Module):
            def __init__(self, layers: List[Dict[str, Any]], input_shape: tuple):
                super().__init__()
                self.layers = nn.ModuleList()
                current_shape = input_shape
                
                for layer_config in layers:
                    if layer_config["type"] == OpType.CONV.value:
                        self.layers.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=current_shape[-1],
                                    out_channels=layer_config["filters"],
                                    kernel_size=layer_config["kernel_size"],
                                    stride=layer_config["stride"],
                                    padding=layer_config["kernel_size"] // 2
                                ),
                                nn.BatchNorm2d(layer_config["filters"]),
                                nn.ReLU() if layer_config["activation"] == "relu"
                                else nn.SiLU()
                            )
                        )
                        current_shape = self.search_space._compute_output_shape(
                            current_shape, layer_config
                        )
                    
                    elif layer_config["type"] == OpType.SEPARABLE_CONV.value:
                        self.layers.append(
                            nn.Sequential(
                                nn.Conv2d(
                                    in_channels=current_shape[-1],
                                    out_channels=current_shape[-1],
                                    kernel_size=layer_config["kernel_size"],
                                    stride=layer_config["stride"],
                                    padding=layer_config["kernel_size"] // 2,
                                    groups=current_shape[-1]
                                ),
                                nn.Conv2d(
                                    in_channels=current_shape[-1],
                                    out_channels=layer_config["filters"],
                                    kernel_size=1
                                ),
                                nn.BatchNorm2d(layer_config["filters"]),
                                nn.ReLU() if layer_config["activation"] == "relu"
                                else nn.SiLU()
                            )
                        )
                        current_shape = self.search_space._compute_output_shape(
                            current_shape, layer_config
                        )
                    
                    elif layer_config["type"] == OpType.ATTENTION.value:
                        self.layers.append(
                            nn.MultiheadAttention(
                                embed_dim=current_shape[-1],
                                num_heads=layer_config["heads"],
                                kdim=layer_config["key_dim"],
                                vdim=layer_config["key_dim"]
                            )
                        )
                    
                    elif layer_config["type"] == "classification":
                        self.layers.append(
                            nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(current_shape[-1], layer_config["n_classes"])
                            )
                        )
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers[:-1]:  # All layers except classification
                    if isinstance(layer, nn.MultiheadAttention):
                        # Reshape for attention
                        b, c, h, w = x.shape
                        x = x.flatten(2).permute(2, 0, 1)  # (L, N, E)
                        x, _ = layer(x, x, x)
                        x = x.permute(1, 2, 0).view(b, c, h, w)
                    else:
                        x = layer(x)
                
                # Classification layer
                return self.layers[-1](x)
        
        return DynamicModel(architecture["layers"], self.search_space.input_shape)

class HardwareEstimator:
    """Estimates hardware requirements for architectures."""
    
    @staticmethod
    def estimate_flops(architecture: Dict[str, Any], input_shape: tuple) -> int:
        """Estimate FLOPs for architecture."""
        total_flops = 0
        current_shape = input_shape
        
        for layer in architecture["layers"]:
            if layer["type"] in [OpType.CONV.value, OpType.SEPARABLE_CONV.value]:
                h, w, in_c = current_shape
                out_c = layer["filters"]
                k = layer["kernel_size"]
                
                if layer["type"] == OpType.CONV.value:
                    # FLOPs = H * W * Cin * Cout * K * K
                    flops = h * w * in_c * out_c * k * k
                else:
                    # Depthwise + Pointwise
                    flops = h * w * in_c * k * k + h * w * in_c * out_c
                
                total_flops += flops
                current_shape = (h//layer["stride"], w//layer["stride"], out_c)
            
            elif layer["type"] == OpType.ATTENTION.value:
                h, w, c = current_shape
                n = h * w  # sequence length
                d = layer["key_dim"]  # key dimension
                h_heads = layer["heads"]
                
                # FLOPs for Q, K, V projections and attention
                flops = 3 * n * c * d + n * n * d * h_heads
                total_flops += flops
        
        return total_flops
    
    @staticmethod
    def estimate_memory(architecture: Dict[str, Any], input_shape: tuple) -> int:
        """Estimate memory requirements in bytes."""
        total_params = 0
        current_shape = input_shape
        
        for layer in architecture["layers"]:
            if layer["type"] in [OpType.CONV.value, OpType.SEPARABLE_CONV.value]:
                h, w, in_c = current_shape
                out_c = layer["filters"]
                k = layer["kernel_size"]
                
                if layer["type"] == OpType.CONV.value:
                    params = out_c * in_c * k * k + out_c  # weights + bias
                else:
                    params = in_c * k * k + out_c * in_c + out_c  # depthwise + pointwise + bias
                
                total_params += params
                current_shape = (h//layer["stride"], w//layer["stride"], out_c)
            
            elif layer["type"] == OpType.ATTENTION.value:
                c = current_shape[-1]
                d = layer["key_dim"]
                h_heads = layer["heads"]
                
                # Parameters for Q, K, V projections
                params = 3 * c * d * h_heads
                total_params += params
        
        # Estimate memory in bytes (assuming float32)
        return total_params * 4
