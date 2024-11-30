from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import redis
import logging
from prometheus_client import Counter, Histogram
from .model_registry import ModelRegistry
from ..config import get_settings

logger = logging.getLogger(__name__)

# Prometheus metrics
EXPERIMENT_REQUESTS = Counter(
    'ab_test_requests_total',
    'Total number of A/B test requests',
    ['experiment_id', 'variant']
)
EXPERIMENT_CONVERSIONS = Counter(
    'ab_test_conversions_total',
    'Total number of A/B test conversions',
    ['experiment_id', 'variant']
)
VARIANT_LATENCY = Histogram(
    'ab_test_latency_seconds',
    'Latency for each variant',
    ['experiment_id', 'variant']
)

class ABTestingManager:
    """Manages A/B testing experiments for model deployment."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            db=0
        )
        self.model_registry = ModelRegistry()
        self.experiments_path = Path("experiments")
        self.experiments_path.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(
        self,
        experiment_id: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_split: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a new A/B testing experiment."""
        try:
            # Validate traffic split
            if traffic_split is None:
                # Equal split between variants
                split_ratio = 1.0 / len(variants)
                traffic_split = {v: split_ratio for v in variants}
            
            if sum(traffic_split.values()) != 1.0:
                raise ValueError("Traffic split must sum to 1.0")
            
            # Create experiment configuration
            experiment = {
                "id": experiment_id,
                "variants": variants,
                "traffic_split": traffic_split,
                "metadata": metadata or {},
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "metrics": {
                    variant: {
                        "requests": 0,
                        "conversions": 0,
                        "latency_sum": 0,
                        "latency_count": 0
                    } for variant in variants
                }
            }
            
            # Save experiment configuration
            self._save_experiment(experiment)
            logger.info(f"Created experiment {experiment_id}")
            
            return experiment
        
        except Exception as e:
            logger.error(f"Failed to create experiment: {str(e)}")
            raise
    
    def _save_experiment(self, experiment: Dict[str, Any]):
        """Save experiment configuration to disk and Redis."""
        experiment_id = experiment["id"]
        
        # Save to disk
        config_path = self.experiments_path / f"{experiment_id}.json"
        with config_path.open("w") as f:
            json.dump(experiment, f, indent=2)
        
        # Save to Redis
        self.redis.set(
            f"experiment:{experiment_id}",
            json.dumps(experiment)
        )
    
    def get_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """Get variant for a user based on traffic split."""
        try:
            # Check if user already has assigned variant
            if user_id:
                variant_key = f"experiment:{experiment_id}:user:{user_id}"
                assigned_variant = self.redis.get(variant_key)
                if assigned_variant:
                    return assigned_variant.decode()
            
            # Get experiment configuration
            experiment = self._get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Select variant based on traffic split
            variants = list(experiment["traffic_split"].keys())
            weights = list(experiment["traffic_split"].values())
            selected_variant = np.random.choice(variants, p=weights)
            
            # Save user assignment if user_id provided
            if user_id:
                self.redis.set(variant_key, selected_variant)
            
            return selected_variant
        
        except Exception as e:
            logger.error(f"Failed to get variant: {str(e)}")
            raise
    
    def record_exposure(
        self,
        experiment_id: str,
        variant: str,
        latency: float
    ):
        """Record exposure to a variant."""
        try:
            # Update Prometheus metrics
            EXPERIMENT_REQUESTS.labels(
                experiment_id=experiment_id,
                variant=variant
            ).inc()
            
            VARIANT_LATENCY.labels(
                experiment_id=experiment_id,
                variant=variant
            ).observe(latency)
            
            # Update experiment metrics
            self._update_metrics(experiment_id, variant, "requests", 1)
            self._update_metrics(experiment_id, variant, "latency_sum", latency)
            self._update_metrics(experiment_id, variant, "latency_count", 1)
        
        except Exception as e:
            logger.error(f"Failed to record exposure: {str(e)}")
            raise
    
    def record_conversion(
        self,
        experiment_id: str,
        variant: str
    ):
        """Record conversion for a variant."""
        try:
            # Update Prometheus metrics
            EXPERIMENT_CONVERSIONS.labels(
                experiment_id=experiment_id,
                variant=variant
            ).inc()
            
            # Update experiment metrics
            self._update_metrics(experiment_id, variant, "conversions", 1)
        
        except Exception as e:
            logger.error(f"Failed to record conversion: {str(e)}")
            raise
    
    def _update_metrics(
        self,
        experiment_id: str,
        variant: str,
        metric: str,
        value: Union[int, float]
    ):
        """Update experiment metrics in Redis."""
        try:
            experiment = self._get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment["metrics"][variant][metric] += value
            self._save_experiment(experiment)
        
        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")
            raise
    
    def get_experiment_results(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """Get experiment results and statistics."""
        try:
            experiment = self._get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            results = {
                "id": experiment_id,
                "variants": {}
            }
            
            for variant, metrics in experiment["metrics"].items():
                requests = metrics["requests"]
                conversions = metrics["conversions"]
                conversion_rate = conversions / requests if requests > 0 else 0
                avg_latency = (
                    metrics["latency_sum"] / metrics["latency_count"]
                    if metrics["latency_count"] > 0 else 0
                )
                
                results["variants"][variant] = {
                    "requests": requests,
                    "conversions": conversions,
                    "conversion_rate": conversion_rate,
                    "avg_latency": avg_latency
                }
            
            return results
        
        except Exception as e:
            logger.error(f"Failed to get experiment results: {str(e)}")
            raise
    
    def _get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment configuration from Redis."""
        experiment_data = self.redis.get(f"experiment:{experiment_id}")
        if experiment_data:
            return json.loads(experiment_data)
        return None
