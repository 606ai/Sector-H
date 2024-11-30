from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import logging
from dataclasses import dataclass
import time
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class MonitoringRule:
    """Represents a monitoring rule for runtime safety."""
    name: str
    condition: Callable
    action: Callable
    priority: int
    description: str

class SafetyMonitor:
    """Runtime safety monitoring system."""
    
    def __init__(
        self,
        rules: List[MonitoringRule],
        buffer_size: int = 1000,
        update_interval: float = 1.0
    ):
        self.rules = sorted(rules, key=lambda x: x.priority)
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Monitoring buffers
        self.input_buffer = deque(maxlen=buffer_size)
        self.output_buffer = deque(maxlen=buffer_size)
        self.violation_buffer = deque(maxlen=buffer_size)
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "violations": 0,
            "last_update": time.time(),
            "rule_stats": {rule.name: {"triggers": 0} for rule in rules}
        }
        
        self.lock = Lock()
    
    def monitor(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, Any]:
        """Monitor input-output pair for safety violations."""
        with self.lock:
            self.stats["total_samples"] += 1
            current_time = time.time()
            
            # Store in buffers
            self.input_buffer.append(x.detach().cpu())
            self.output_buffer.append(y.detach().cpu())
            
            # Check rules
            violations = []
            actions_taken = []
            
            for rule in self.rules:
                try:
                    if rule.condition(x, y):
                        violations.append(rule.name)
                        self.stats["rule_stats"][rule.name]["triggers"] += 1
                        
                        # Execute action
                        action_result = rule.action(x, y)
                        actions_taken.append({
                            "rule": rule.name,
                            "result": action_result
                        })
                except Exception as e:
                    logger.error(f"Error in rule {rule.name}: {str(e)}")
            
            if violations:
                self.stats["violations"] += 1
                self.violation_buffer.append({
                    "timestamp": current_time,
                    "violations": violations,
                    "actions": actions_taken
                })
            
            # Update statistics if interval has passed
            if current_time - self.stats["last_update"] >= self.update_interval:
                self._update_statistics()
                self.stats["last_update"] = current_time
            
            return {
                "violations": violations,
                "actions": actions_taken
            }
    
    def _update_statistics(self):
        """Update monitoring statistics."""
        if not self.input_buffer:
            return
        
        # Compute basic statistics
        recent_inputs = torch.stack(list(self.input_buffer))
        recent_outputs = torch.stack(list(self.output_buffer))
        
        self.stats.update({
            "input_mean": recent_inputs.mean().item(),
            "input_std": recent_inputs.std().item(),
            "output_mean": recent_outputs.mean().item(),
            "output_std": recent_outputs.std().item(),
            "violation_rate": self.stats["violations"] / self.stats["total_samples"]
        })
        
        # Update rule-specific statistics
        for rule in self.rules:
            rule_stats = self.stats["rule_stats"][rule.name]
            rule_stats["trigger_rate"] = (
                rule_stats["triggers"] / self.stats["total_samples"]
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        with self.lock:
            return self.stats.copy()
    
    def get_recent_violations(
        self,
        n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent safety violations."""
        with self.lock:
            violations = list(self.violation_buffer)
            if n is not None:
                violations = violations[-n:]
            return violations
    
    def clear_buffers(self):
        """Clear monitoring buffers."""
        with self.lock:
            self.input_buffer.clear()
            self.output_buffer.clear()
            self.violation_buffer.clear()

class DistributionMonitor:
    """Monitors distribution shifts in input/output spaces."""
    
    def __init__(
        self,
        window_size: int = 1000,
        threshold: float = 0.1
    ):
        self.window_size = window_size
        self.threshold = threshold
        
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        
        self.stats = {
            "distribution_shifts": 0,
            "last_shift_time": None,
            "current_divergence": 0.0
        }
    
    def update_reference(
        self,
        samples: torch.Tensor
    ):
        """Update reference distribution window."""
        for sample in samples:
            self.reference_window.append(sample.detach().cpu())
    
    def monitor(
        self,
        sample: torch.Tensor
    ) -> Dict[str, Any]:
        """Monitor for distribution shifts."""
        self.current_window.append(sample.detach().cpu())
        
        if len(self.current_window) < self.window_size:
            return {"shift_detected": False}
        
        # Compute distribution divergence
        divergence = self._compute_divergence(
            torch.stack(list(self.reference_window)),
            torch.stack(list(self.current_window))
        )
        
        self.stats["current_divergence"] = divergence
        
        # Check for shift
        shift_detected = divergence > self.threshold
        
        if shift_detected:
            self.stats["distribution_shifts"] += 1
            self.stats["last_shift_time"] = time.time()
        
        return {
            "shift_detected": shift_detected,
            "divergence": divergence
        }
    
    def _compute_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor
    ) -> float:
        """Compute distribution divergence (MMD)."""
        # Compute kernel matrices
        xx = torch.mm(p, p.t())
        xy = torch.mm(p, q.t())
        yy = torch.mm(q, q.t())
        
        # Compute MMD
        dx = p.size(0)
        dy = q.size(0)
        
        mmd = (xx.sum() / (dx * dx) + 
               yy.sum() / (dy * dy) - 
               2 * xy.sum() / (dx * dy))
        
        return mmd.item()

class GradientMonitor:
    """Monitors gradient behavior during training."""
    
    def __init__(
        self,
        clip_threshold: float = 1.0,
        window_size: int = 100
    ):
        self.clip_threshold = clip_threshold
        self.window_size = window_size
        
        self.grad_history = deque(maxlen=window_size)
        self.stats = {
            "clipped_gradients": 0,
            "total_gradients": 0,
            "max_gradient_norm": 0.0
        }
    
    def monitor(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Monitor gradients of model parameters."""
        grad_norms = []
        clipped = 0
        total = 0
        
        for param in model.parameters():
            if param.grad is not None:
                total += 1
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if grad_norm > self.clip_threshold:
                    clipped += 1
        
        if grad_norms:
            max_norm = max(grad_norms)
            self.grad_history.append(max_norm)
            
            self.stats["clipped_gradients"] += clipped
            self.stats["total_gradients"] += total
            self.stats["max_gradient_norm"] = max(
                self.stats["max_gradient_norm"],
                max_norm
            )
        
        return {
            "clipped_ratio": clipped / total if total > 0 else 0.0,
            "max_norm": max_norm if grad_norms else 0.0,
            "mean_norm": np.mean(grad_norms) if grad_norms else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gradient monitoring statistics."""
        return {
            "clip_ratio": (self.stats["clipped_gradients"] / 
                          self.stats["total_gradients"]
                          if self.stats["total_gradients"] > 0 else 0.0),
            "max_gradient": self.stats["max_gradient_norm"],
            "recent_gradients": list(self.grad_history)
        }

class MonitoringDashboard:
    """Dashboard for visualizing safety monitoring results."""
    
    def __init__(
        self,
        safety_monitor: SafetyMonitor,
        dist_monitor: Optional[DistributionMonitor] = None,
        grad_monitor: Optional[GradientMonitor] = None,
        update_interval: float = 1.0
    ):
        self.safety_monitor = safety_monitor
        self.dist_monitor = dist_monitor
        self.grad_monitor = grad_monitor
        self.update_interval = update_interval
        
        self.last_update = time.time()
        self.dashboard_data = {}
    
    def update(self) -> Dict[str, Any]:
        """Update dashboard with latest monitoring data."""
        current_time = time.time()
        
        if current_time - self.last_update < self.update_interval:
            return self.dashboard_data
        
        # Gather data from all monitors
        self.dashboard_data = {
            "safety": self.safety_monitor.get_statistics(),
            "recent_violations": self.safety_monitor.get_recent_violations(10)
        }
        
        if self.dist_monitor:
            self.dashboard_data["distribution"] = {
                "shifts": self.dist_monitor.stats["distribution_shifts"],
                "current_divergence": self.dist_monitor.stats["current_divergence"],
                "last_shift": self.dist_monitor.stats["last_shift_time"]
            }
        
        if self.grad_monitor:
            self.dashboard_data["gradients"] = self.grad_monitor.get_statistics()
        
        self.last_update = current_time
        return self.dashboard_data
    
    def get_summary(self) -> str:
        """Get text summary of monitoring status."""
        data = self.update()
        
        summary = [
            "=== Safety Monitoring Summary ===",
            f"Total Samples: {data['safety']['total_samples']}",
            f"Violation Rate: {data['safety']['violation_rate']:.2%}",
            "\nRecent Violations:"
        ]
        
        for violation in data["recent_violations"]:
            summary.append(
                f"- {time.strftime('%H:%M:%S', time.localtime(violation['timestamp']))}: "
                f"{', '.join(violation['violations'])}"
            )
        
        if "distribution" in data:
            summary.extend([
                "\nDistribution Monitoring:",
                f"Total Shifts: {data['distribution']['shifts']}",
                f"Current Divergence: {data['distribution']['current_divergence']:.4f}"
            ])
        
        if "gradients" in data:
            summary.extend([
                "\nGradient Monitoring:",
                f"Clip Ratio: {data['gradients']['clip_ratio']:.2%}",
                f"Max Gradient: {data['gradients']['max_gradient']:.4f}"
            ])
        
        return "\n".join(summary)
