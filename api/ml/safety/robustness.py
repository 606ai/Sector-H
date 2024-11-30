from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RobustnessMetric:
    """Represents a robustness metric for evaluation."""
    name: str
    threshold: float
    description: str

class RobustnessTester:
    """Tests model robustness against various perturbations."""
    
    def __init__(
        self,
        metrics: List[RobustnessMetric],
        epsilon: float = 0.1,
        n_samples: int = 1000
    ):
        self.metrics = metrics
        self.epsilon = epsilon
        self.n_samples = n_samples
    
    def test_gaussian_noise(
        self,
        model: nn.Module,
        x: torch.Tensor,
        std: float = 0.1
    ) -> Dict[str, float]:
        """Test robustness against Gaussian noise."""
        results = {}
        
        # Generate noisy samples
        noise = torch.randn_like(x) * std
        x_noisy = x + noise
        
        # Get predictions
        with torch.no_grad():
            y = model(x)
            y_noisy = model(x_noisy)
        
        # Compute metrics
        for metric in self.metrics:
            if metric.name == "l2_stability":
                value = torch.norm(y - y_noisy) / torch.norm(noise)
            elif metric.name == "prediction_stability":
                value = torch.mean((torch.argmax(y, dim=1) == 
                                  torch.argmax(y_noisy, dim=1)).float())
            else:
                continue
            
            results[metric.name] = value.item()
            results[f"{metric.name}_passed"] = value.item() <= metric.threshold
        
        return results
    
    def test_adversarial(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y_true: torch.Tensor,
        attack_fn: Callable
    ) -> Dict[str, float]:
        """Test robustness against adversarial attacks."""
        results = {}
        
        # Generate adversarial examples
        x_adv = attack_fn(model, x, y_true)
        
        # Get predictions
        with torch.no_grad():
            y = model(x)
            y_adv = model(x_adv)
        
        # Compute metrics
        for metric in self.metrics:
            if metric.name == "adversarial_accuracy":
                value = torch.mean((torch.argmax(y_adv, dim=1) == 
                                  torch.argmax(y_true, dim=1)).float())
            elif metric.name == "perturbation_size":
                value = torch.norm(x - x_adv) / torch.norm(x)
            else:
                continue
            
            results[metric.name] = value.item()
            results[f"{metric.name}_passed"] = value.item() <= metric.threshold
        
        return results
    
    def test_interpolation(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        x2: torch.Tensor,
        n_points: int = 100
    ) -> Dict[str, float]:
        """Test robustness along interpolated paths."""
        results = {}
        
        # Generate interpolated points
        alphas = torch.linspace(0, 1, n_points)
        x_interp = torch.stack([
            alpha * x1 + (1 - alpha) * x2
            for alpha in alphas
        ])
        
        # Get predictions
        with torch.no_grad():
            y_interp = model(x_interp)
        
        # Compute metrics
        for metric in self.metrics:
            if metric.name == "lipschitz_estimate":
                diffs = torch.norm(y_interp[1:] - y_interp[:-1], dim=1)
                value = torch.max(diffs / torch.norm(x_interp[1:] - x_interp[:-1], dim=1))
            elif metric.name == "smoothness":
                diffs = torch.norm(y_interp[2:] - 2*y_interp[1:-1] + y_interp[:-2], dim=1)
                value = torch.mean(diffs)
            else:
                continue
            
            results[metric.name] = value.item()
            results[f"{metric.name}_passed"] = value.item() <= metric.threshold
        
        return results
    
    def test_distribution_shift(
        self,
        model: nn.Module,
        x: torch.Tensor,
        shift_fn: Callable
    ) -> Dict[str, float]:
        """Test robustness under distribution shift."""
        results = {}
        
        # Generate shifted samples
        x_shifted = shift_fn(x)
        
        # Get predictions
        with torch.no_grad():
            y = model(x)
            y_shifted = model(x_shifted)
        
        # Compute metrics
        for metric in self.metrics:
            if metric.name == "wasserstein_distance":
                value = self._compute_wasserstein(y, y_shifted)
            elif metric.name == "kl_divergence":
                value = self._compute_kl_divergence(y, y_shifted)
            else:
                continue
            
            results[metric.name] = value.item()
            results[f"{metric.name}_passed"] = value.item() <= metric.threshold
        
        return results
    
    def _compute_wasserstein(
        self,
        p: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """Compute Wasserstein distance between distributions."""
        # Sort the distributions
        p_sorted, _ = torch.sort(p, dim=1)
        q_sorted, _ = torch.sort(q, dim=1)
        
        # Compute L2 Wasserstein distance
        return torch.mean(torch.sqrt(torch.sum((p_sorted - q_sorted)**2, dim=1)))
    
    def _compute_kl_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / torch.sum(p, dim=1, keepdim=True)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return torch.mean(torch.sum(p * torch.log(p / q), dim=1))

class CertifiedRobustness:
    """Provides certified robustness guarantees."""
    
    def __init__(
        self,
        sigma: float = 0.25,
        n_samples: int = 1000,
        alpha: float = 0.001
    ):
        self.sigma = sigma
        self.n_samples = n_samples
        self.alpha = alpha
    
    def certify(
        self,
        model: nn.Module,
        x: torch.Tensor,
        n_classes: int
    ) -> Tuple[int, float]:
        """Certify robustness using randomized smoothing."""
        # Generate noise samples
        noise = torch.randn(
            (self.n_samples,) + x.shape,
            device=x.device
        ) * self.sigma
        
        x_noisy = x.unsqueeze(0) + noise
        
        # Get predictions
        with torch.no_grad():
            predictions = model(x_noisy)
            counts = torch.zeros(n_classes, device=x.device)
            
            for pred in predictions:
                counts[torch.argmax(pred)] += 1
            
            # Get top two classes
            top_class = torch.argmax(counts)
            counts[top_class] = -1
            runner_up = torch.argmax(counts)
            
            # Compute radius using Gaussian CDF
            p_a = counts[top_class].item() / self.n_samples
            p_b = counts[runner_up].item() / self.n_samples
            
            if p_a < 0.5:
                return -1, 0.0
            
            radius = self.sigma * (norm.ppf(p_a) - norm.ppf(p_b)) / 2
            
            return top_class.item(), radius

class RobustnessReport:
    """Generates comprehensive robustness reports."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        tester: RobustnessTester,
        certifier: Optional[CertifiedRobustness] = None
    ):
        self.model = model
        self.test_loader = test_loader
        self.tester = tester
        self.certifier = certifier
        self.results: Dict[str, Any] = {}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive robustness report."""
        self.results = {
            "gaussian_noise": [],
            "adversarial": [],
            "interpolation": [],
            "distribution_shift": []
        }
        
        if self.certifier is not None:
            self.results["certified"] = []
        
        # Test on dataset
        for x, y in self.test_loader:
            # Basic tests
            noise_results = self.tester.test_gaussian_noise(self.model, x)
            self.results["gaussian_noise"].append(noise_results)
            
            # Adversarial test (using PGD attack as example)
            adv_results = self.tester.test_adversarial(
                self.model, x, y,
                self._pgd_attack
            )
            self.results["adversarial"].append(adv_results)
            
            # Interpolation test
            if len(x) >= 2:
                interp_results = self.tester.test_interpolation(
                    self.model, x[0], x[1]
                )
                self.results["interpolation"].append(interp_results)
            
            # Distribution shift test
            shift_results = self.tester.test_distribution_shift(
                self.model, x,
                self._apply_shift
            )
            self.results["distribution_shift"].append(shift_results)
            
            # Certified robustness
            if self.certifier is not None:
                cert_class, radius = self.certifier.certify(
                    self.model, x[0],
                    self.model.output_size
                )
                self.results["certified"].append({
                    "class": cert_class,
                    "radius": radius
                })
        
        # Aggregate results
        return self._aggregate_results()
    
    def _pgd_attack(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.3,
        alpha: float = 0.01,
        num_iter: int = 40
    ) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        delta = torch.zeros_like(x, requires_grad=True)
        
        for _ in range(num_iter):
            output = model(x + delta)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            
            delta.data = (delta + alpha * delta.grad.sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        
        return x + delta.detach()
    
    def _apply_shift(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply simple distribution shift."""
        return x + torch.randn_like(x) * 0.1
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate test results."""
        aggregated = {}
        
        for test_type, results in self.results.items():
            if test_type == "certified":
                # Aggregate certification results
                valid_radii = [r["radius"] for r in results if r["class"] != -1]
                aggregated[test_type] = {
                    "mean_radius": np.mean(valid_radii),
                    "certified_ratio": len(valid_radii) / len(results)
                }
            else:
                # Aggregate other test results
                test_metrics = {}
                for metric in self.tester.metrics:
                    values = [r[metric.name] for r in results]
                    passed = [r[f"{metric.name}_passed"] for r in results]
                    
                    test_metrics[metric.name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "pass_rate": np.mean(passed)
                    }
                
                aggregated[test_type] = test_metrics
        
        return aggregated
