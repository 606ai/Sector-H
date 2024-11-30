from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SafetyProperty:
    """Represents a safety property to verify."""
    name: str
    input_constraints: Dict[str, Tuple[float, float]]
    output_constraints: Dict[str, Tuple[float, float]]
    description: str

class SafetyVerifier(ABC):
    """Base class for neural network safety verification."""
    
    @abstractmethod
    def verify(
        self,
        model: nn.Module,
        property: SafetyProperty
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify if a model satisfies a safety property."""
        pass

class LipschitzVerifier(SafetyVerifier):
    """Verifier using Lipschitz continuity."""
    
    def __init__(
        self,
        epsilon: float = 1e-3,
        max_iter: int = 1000
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def verify(
        self,
        model: nn.Module,
        property: SafetyProperty
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify using Lipschitz constant estimation."""
        try:
            # Compute Lipschitz constant
            lipschitz_constant = self._estimate_lipschitz(model)
            
            # Check if property is satisfied
            input_range = max(
                high - low
                for low, high in property.input_constraints.values()
            )
            output_range = min(
                high - low
                for low, high in property.output_constraints.values()
            )
            
            # Property is satisfied if Lipschitz constant * input_range <= output_range
            is_safe = lipschitz_constant * input_range <= output_range
            
            info = {
                "lipschitz_constant": lipschitz_constant,
                "input_range": input_range,
                "output_range": output_range,
                "margin": output_range - lipschitz_constant * input_range
            }
            
            return is_safe, info
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False, None
    
    def _estimate_lipschitz(
        self,
        model: nn.Module
    ) -> float:
        """Estimate Lipschitz constant of the model."""
        # Power iteration method
        with torch.no_grad():
            v = torch.randn(1, model.input_size)
            v = v / torch.norm(v)
            
            for _ in range(self.max_iter):
                # Forward pass
                v.requires_grad_(True)
                y = model(v)
                
                # Compute gradient
                grad = torch.autograd.grad(
                    y.sum(),
                    v,
                    create_graph=True
                )[0]
                
                # Update direction
                v = grad / torch.norm(grad)
                
                if torch.norm(grad) < self.epsilon:
                    break
            
            return torch.norm(grad).item()

class BoundVerifier(SafetyVerifier):
    """Verifier using interval bound propagation."""
    
    def verify(
        self,
        model: nn.Module,
        property: SafetyProperty
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify using interval bound propagation."""
        try:
            # Initialize bounds
            lower_bounds = []
            upper_bounds = []
            
            # Input bounds
            input_lower = torch.tensor([
                low for low, _ in property.input_constraints.values()
            ])
            input_upper = torch.tensor([
                high for _, high in property.input_constraints.values()
            ])
            
            lower_bounds.append(input_lower)
            upper_bounds.append(input_upper)
            
            # Propagate bounds through layers
            for layer in model.children():
                if isinstance(layer, nn.Linear):
                    l_bound, u_bound = self._propagate_linear(
                        layer,
                        lower_bounds[-1],
                        upper_bounds[-1]
                    )
                elif isinstance(layer, nn.ReLU):
                    l_bound, u_bound = self._propagate_relu(
                        lower_bounds[-1],
                        upper_bounds[-1]
                    )
                else:
                    raise ValueError(f"Unsupported layer type: {type(layer)}")
                
                lower_bounds.append(l_bound)
                upper_bounds.append(u_bound)
            
            # Check if output bounds satisfy constraints
            output_lower = lower_bounds[-1]
            output_upper = upper_bounds[-1]
            
            is_safe = True
            for i, (low, high) in enumerate(
                property.output_constraints.values()
            ):
                if output_lower[i] < low or output_upper[i] > high:
                    is_safe = False
                    break
            
            info = {
                "input_bounds": (input_lower, input_upper),
                "output_bounds": (output_lower, output_upper),
                "layer_bounds": list(zip(lower_bounds, upper_bounds))
            }
            
            return is_safe, info
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False, None
    
    def _propagate_linear(
        self,
        layer: nn.Linear,
        lower: torch.Tensor,
        upper: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through linear layer."""
        weight = layer.weight
        bias = layer.bias if layer.bias is not None else 0
        
        # Compute output bounds
        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)
        
        lower_bound = (
            weight_pos @ lower +
            weight_neg @ upper +
            bias
        )
        upper_bound = (
            weight_pos @ upper +
            weight_neg @ lower +
            bias
        )
        
        return lower_bound, upper_bound
    
    def _propagate_relu(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate bounds through ReLU layer."""
        return torch.clamp(lower, min=0), torch.clamp(upper, min=0)

class AdversarialVerifier(SafetyVerifier):
    """Verifier using adversarial attacks."""
    
    def __init__(
        self,
        n_attempts: int = 100,
        step_size: float = 0.01,
        max_iter: int = 100
    ):
        self.n_attempts = n_attempts
        self.step_size = step_size
        self.max_iter = max_iter
    
    def verify(
        self,
        model: nn.Module,
        property: SafetyProperty
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify by searching for adversarial examples."""
        try:
            # Try to find counterexample
            best_violation = 0.0
            best_input = None
            
            for _ in range(self.n_attempts):
                # Random starting point
                x = torch.tensor([
                    np.random.uniform(low, high)
                    for (low, high) in property.input_constraints.values()
                ], requires_grad=True)
                
                # Gradient descent to find violation
                optimizer = torch.optim.Adam([x], lr=self.step_size)
                
                for _ in range(self.max_iter):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    y = model(x.unsqueeze(0)).squeeze(0)
                    
                    # Compute violation
                    violation = 0.0
                    for i, (low, high) in enumerate(
                        property.output_constraints.values()
                    ):
                        violation += torch.relu(low - y[i])
                        violation += torch.relu(y[i] - high)
                    
                    if violation.item() > best_violation:
                        best_violation = violation.item()
                        best_input = x.detach().clone()
                    
                    # Update
                    violation.backward()
                    optimizer.step()
                    
                    # Project back to input constraints
                    with torch.no_grad():
                        for i, (low, high) in enumerate(
                            property.input_constraints.values()
                        ):
                            x[i].clamp_(low, high)
            
            is_safe = best_violation <= 1e-6
            
            info = {
                "violation": best_violation,
                "counterexample": best_input.numpy() if best_input is not None else None,
                "n_attempts": self.n_attempts,
                "max_iter": self.max_iter
            }
            
            return is_safe, info
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False, None

class SafetyMonitor:
    """Runtime safety monitor for neural networks."""
    
    def __init__(
        self,
        properties: List[SafetyProperty],
        verifiers: List[SafetyVerifier]
    ):
        self.properties = properties
        self.verifiers = verifiers
        self.verification_results: List[Dict[str, Any]] = []
    
    def verify_model(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Verify model against all safety properties."""
        results = {}
        
        for prop in self.properties:
            prop_results = {}
            
            for verifier in self.verifiers:
                is_safe, info = verifier.verify(model, prop)
                prop_results[verifier.__class__.__name__] = {
                    "is_safe": is_safe,
                    "info": info
                }
            
            results[prop.name] = prop_results
        
        self.verification_results.append(results)
        return results
    
    def monitor_input(
        self,
        x: torch.Tensor
    ) -> bool:
        """Check if input satisfies all input constraints."""
        for prop in self.properties:
            for i, (low, high) in enumerate(
                prop.input_constraints.values()
            ):
                if x[i] < low or x[i] > high:
                    return False
        return True
    
    def monitor_output(
        self,
        y: torch.Tensor
    ) -> bool:
        """Check if output satisfies all output constraints."""
        for prop in self.properties:
            for i, (low, high) in enumerate(
                prop.output_constraints.values()
            ):
                if y[i] < low or y[i] > high:
                    return False
        return True
    
    def get_verification_history(self) -> List[Dict[str, Any]]:
        """Get history of verification results."""
        return self.verification_results
