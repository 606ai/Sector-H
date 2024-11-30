from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class EnsembleUncertainty:
    """Uncertainty estimation using ensemble methods."""
    
    def __init__(
        self,
        models: List[nn.Module],
        temperature: float = 1.0
    ):
        self.models = models
        self.temperature = temperature
    
    def predict(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Get ensemble predictions and uncertainty estimates."""
        predictions = []
        
        # Get predictions from each model
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = F.softmax(
                    model(x) / self.temperature,
                    dim=-1
                )
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute mean prediction
        mean_pred = torch.mean(predictions, dim=0)
        
        # Compute uncertainty metrics
        entropy_uncertainty = self._compute_entropy(mean_pred)
        mutual_info = self._compute_mutual_information(predictions)
        variance = torch.var(predictions, dim=0)
        
        result = {
            "mean_prediction": mean_pred,
            "entropy": entropy_uncertainty,
            "mutual_information": mutual_info,
            "variance": variance
        }
        
        if return_individual:
            result["individual_predictions"] = predictions
        
        return result
    
    def _compute_entropy(
        self,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy of predictions."""
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    def _compute_mutual_information(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """Compute mutual information (predictive entropy - expected entropy)."""
        # Compute predictive entropy
        mean_pred = torch.mean(predictions, dim=0)
        predictive_entropy = self._compute_entropy(mean_pred)
        
        # Compute expected entropy
        expected_entropy = torch.mean(
            self._compute_entropy(predictions),
            dim=0
        )
        
        return predictive_entropy - expected_entropy

class BayesianUncertainty:
    """Uncertainty estimation using Bayesian methods."""
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100
    ):
        self.model = model
        self.n_samples = n_samples
    
    def enable_dropout(
        self,
        model: nn.Module
    ):
        """Enable dropout at inference time."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def predict(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get Bayesian predictions and uncertainty estimates."""
        self.model.eval()
        self.enable_dropout(self.model)
        
        predictions = []
        
        # Multiple forward passes with dropout
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = F.softmax(self.model(x), dim=-1)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute mean and uncertainty
        mean_pred = torch.mean(predictions, dim=0)
        aleatoric = self._compute_aleatoric(predictions)
        epistemic = self._compute_epistemic(predictions, mean_pred)
        
        return {
            "mean_prediction": mean_pred,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_uncertainty": aleatoric + epistemic
        }
    
    def _compute_aleatoric(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """Compute aleatoric uncertainty."""
        return torch.mean(
            predictions * (1 - predictions),
            dim=0
        )
    
    def _compute_epistemic(
        self,
        predictions: torch.Tensor,
        mean_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute epistemic uncertainty."""
        return torch.mean(
            (predictions - mean_pred.unsqueeze(0))**2,
            dim=0
        )

class EvidentialUncertainty:
    """Uncertainty estimation using evidential learning."""
    
    def __init__(
        self,
        n_classes: int,
        epsilon: float = 1e-10
    ):
        self.n_classes = n_classes
        self.epsilon = epsilon
    
    def compute_evidence(
        self,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute Dirichlet evidence from network output."""
        # Convert logits to evidence
        evidence = F.relu(logits)
        
        # Compute Dirichlet parameters
        alpha = evidence + 1
        
        # Compute uncertainty metrics
        strength = torch.sum(alpha, dim=-1)
        uncertainty = self.n_classes / strength
        
        # Compute expected probability
        prob = alpha / strength.unsqueeze(-1)
        
        return {
            "evidence": evidence,
            "alpha": alpha,
            "uncertainty": uncertainty,
            "probability": prob
        }
    
    def edl_loss(
        self,
        evidence: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute evidential learning loss."""
        alpha = evidence + 1
        strength = torch.sum(alpha, dim=-1)
        
        # Compute loss terms
        loss_term1 = torch.sum(
            targets * (torch.digamma(strength.unsqueeze(-1)) - 
                      torch.digamma(alpha)),
            dim=-1
        )
        
        loss_term2 = torch.lgamma(strength) - torch.sum(
            torch.lgamma(alpha),
            dim=-1
        )
        
        return torch.mean(loss_term1 + loss_term2)

class ConformalPrediction:
    """Uncertainty estimation using conformal prediction."""
    
    def __init__(
        self,
        alpha: float = 0.1,
        calibration_size: int = 1000
    ):
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.calibration_scores = None
    
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ):
        """Calibrate conformal predictor using validation data."""
        scores = []
        
        model.eval()
        with torch.no_grad():
            for x, y in calibration_data:
                # Get model predictions
                pred = model(x)
                
                # Compute conformity scores
                scores.append(self._compute_conformity(pred, y))
        
        # Combine all scores
        self.calibration_scores = torch.cat(scores)
        
        # Sort scores and find threshold
        sorted_scores, _ = torch.sort(self.calibration_scores)
        idx = int((1 - self.alpha) * len(sorted_scores))
        self.threshold = sorted_scores[idx]
    
    def predict(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> Dict[str, Any]:
        """Get predictions with conformal prediction sets."""
        if self.calibration_scores is None:
            raise ValueError("Calibrate the predictor first")
        
        model.eval()
        with torch.no_grad():
            # Get model predictions
            pred = model(x)
            
            # Compute prediction sets
            pred_sets = self._compute_prediction_sets(pred)
            
            return {
                "prediction": pred,
                "prediction_sets": pred_sets,
                "set_sizes": torch.sum(pred_sets, dim=-1)
            }
    
    def _compute_conformity(
        self,
        pred: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute conformity scores."""
        sorted_pred, _ = torch.sort(pred, dim=-1, descending=True)
        true_scores = torch.gather(pred, -1, y.unsqueeze(-1))
        
        return -true_scores.squeeze(-1)
    
    def _compute_prediction_sets(
        self,
        pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute conformal prediction sets."""
        sorted_pred, indices = torch.sort(pred, dim=-1, descending=True)
        cumsum_pred = torch.cumsum(sorted_pred, dim=-1)
        
        # Find smallest set satisfying coverage
        set_sizes = torch.sum(
            cumsum_pred <= (1 - self.threshold).unsqueeze(-1),
            dim=-1
        )
        
        # Create prediction sets
        pred_sets = torch.zeros_like(pred, dtype=torch.bool)
        for i in range(len(pred)):
            pred_sets[i, indices[i, :set_sizes[i]]] = True
        
        return pred_sets

class UncertaintyMetrics:
    """Compute and track uncertainty metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None
    ):
        """Update uncertainty metrics."""
        # Compute basic uncertainty metrics
        self.metrics.update({
            "mean_entropy": torch.mean(
                self._entropy(predictions["mean_prediction"])
            ).item(),
            "mean_variance": torch.mean(
                predictions.get("variance", torch.tensor(0.))
            ).item()
        })
        
        # Compute calibration metrics if targets available
        if targets is not None:
            calibration = self._compute_calibration(
                predictions["mean_prediction"],
                targets
            )
            self.metrics.update(calibration)
        
        # Compute additional uncertainty metrics if available
        if "aleatoric" in predictions:
            self.metrics["mean_aleatoric"] = torch.mean(
                predictions["aleatoric"]
            ).item()
            self.metrics["mean_epistemic"] = torch.mean(
                predictions["epistemic"]
            ).item()
        
        if "mutual_information" in predictions:
            self.metrics["mean_mutual_info"] = torch.mean(
                predictions["mutual_information"]
            ).item()
    
    def _entropy(
        self,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy of predictions."""
        return -torch.sum(
            probs * torch.log(probs + 1e-10),
            dim=-1
        )
    
    def _compute_calibration(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute calibration metrics."""
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = (predictions == targets).float()
        
        # Compute ECE
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
        
        return {
            "ece": ece.item(),
            "accuracy": accuracies.mean().item(),
            "mean_confidence": confidences.mean().item()
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current uncertainty metrics."""
        return self.metrics.copy()

class UncertaintyCallback:
    """Callback for monitoring uncertainty during training."""
    
    def __init__(
        self,
        uncertainty_estimator: Any,
        metrics: UncertaintyMetrics,
        eval_interval: int = 100
    ):
        self.uncertainty_estimator = uncertainty_estimator
        self.metrics = metrics
        self.eval_interval = eval_interval
        self.step = 0
        self.history = []
    
    def __call__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ):
        """Update uncertainty metrics during training."""
        self.step += 1
        
        if self.step % self.eval_interval == 0:
            # Compute uncertainty estimates
            predictions = self.uncertainty_estimator.predict(x)
            
            # Update metrics
            self.metrics.update(predictions, y)
            
            # Store history
            self.history.append({
                "step": self.step,
                "metrics": self.metrics.get_metrics()
            })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get uncertainty metric history."""
        return self.history
