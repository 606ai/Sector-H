from typing import Dict, List, Set, Optional, Tuple, Any
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
import json

logger = logging.getLogger(__name__)

@dataclass
class CausalVariable:
    """Represents a variable in the causal graph."""
    name: str
    type: str  # 'continuous', 'categorical', 'binary'
    parents: Set[str]
    children: Set[str]
    observed: bool = True
    domain: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None

class StructuralCausalModel:
    """Structural Causal Model (SCM) implementation."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.variables: Dict[str, CausalVariable] = {}
        self.mechanisms: Dict[str, nn.Module] = {}
        self.interventions: Dict[str, Any] = {}
    
    def add_variable(
        self,
        name: str,
        var_type: str,
        parents: Optional[Set[str]] = None,
        observed: bool = True,
        domain: Optional[Tuple[float, float]] = None,
        categories: Optional[List[str]] = None
    ):
        """Add a variable to the causal graph."""
        parents = parents or set()
        children = set()
        
        # Update children of parent nodes
        for parent in parents:
            if parent in self.variables:
                self.variables[parent].children.add(name)
        
        # Create variable
        var = CausalVariable(
            name=name,
            type=var_type,
            parents=parents,
            children=children,
            observed=observed,
            domain=domain,
            categories=categories
        )
        
        self.variables[name] = var
        self.graph.add_node(name)
        
        # Add edges from parents
        for parent in parents:
            self.graph.add_edge(parent, name)
    
    def add_mechanism(
        self,
        variable: str,
        mechanism: nn.Module
    ):
        """Add a causal mechanism for a variable."""
        if variable not in self.variables:
            raise ValueError(f"Variable {variable} not in model")
        
        self.mechanisms[variable] = mechanism
    
    def do_intervention(
        self,
        interventions: Dict[str, Any]
    ):
        """Perform interventions on variables."""
        self.interventions.update(interventions)
    
    def reset_interventions(self):
        """Reset all interventions."""
        self.interventions.clear()
    
    def sample(
        self,
        n_samples: int,
        include_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Sample from the structural causal model."""
        samples = {}
        ordered_vars = list(nx.topological_sort(self.graph))
        
        for var_name in ordered_vars:
            var = self.variables[var_name]
            
            # Skip unobserved variables unless specifically requested
            if not var.observed and not include_latent:
                continue
            
            # Check for intervention
            if var_name in self.interventions:
                value = self.interventions[var_name]
                if isinstance(value, (int, float)):
                    samples[var_name] = torch.full((n_samples,), value)
                else:
                    samples[var_name] = torch.tensor(value).expand(n_samples, -1)
                continue
            
            # Get parent values
            parent_values = []
            for parent in var.parents:
                if parent in samples:
                    parent_values.append(samples[parent])
            
            # Generate samples using mechanism
            if var_name in self.mechanisms:
                mechanism = self.mechanisms[var_name]
                parent_tensor = torch.cat(parent_values, dim=1) if parent_values else torch.empty(n_samples, 0)
                samples[var_name] = mechanism(parent_tensor)
            else:
                # Default sampling if no mechanism specified
                if var.type == 'continuous':
                    if var.domain:
                        samples[var_name] = torch.rand(n_samples) * (var.domain[1] - var.domain[0]) + var.domain[0]
                    else:
                        samples[var_name] = torch.randn(n_samples)
                elif var.type == 'categorical':
                    n_categories = len(var.categories) if var.categories else 2
                    samples[var_name] = torch.randint(0, n_categories, (n_samples,))
                elif var.type == 'binary':
                    samples[var_name] = torch.randint(0, 2, (n_samples,))
        
        return samples
    
    def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        n_samples: int = 1000,
        method: str = 'intervention'
    ) -> Dict[str, float]:
        """Estimate causal effect between treatment and outcome."""
        if method == 'intervention':
            # Estimate effect through intervention
            baseline_samples = self.sample(n_samples)
            baseline_outcome = baseline_samples[outcome].mean().item()
            
            # Intervene on treatment
            if self.variables[treatment].type == 'continuous':
                intervention_value = 1.0
            else:
                intervention_value = 1
            
            self.do_intervention({treatment: intervention_value})
            intervention_samples = self.sample(n_samples)
            intervention_outcome = intervention_samples[outcome].mean().item()
            
            self.reset_interventions()
            
            effect = intervention_outcome - baseline_outcome
            
        elif method == 'backdoor':
            # Implement backdoor adjustment
            backdoor_set = self._find_backdoor_set(treatment, outcome)
            effect = self._backdoor_adjustment(
                treatment,
                outcome,
                backdoor_set,
                n_samples
            )
        
        else:
            raise ValueError(f"Unknown estimation method: {method}")
        
        return {
            "effect": effect,
            "method": method,
            "n_samples": n_samples
        }
    
    def _find_backdoor_set(
        self,
        treatment: str,
        outcome: str
    ) -> Set[str]:
        """Find a valid backdoor adjustment set."""
        # Implement Pearl's backdoor criterion
        backdoor_set = set()
        
        # Get ancestors of treatment and outcome
        treatment_ancestors = nx.ancestors(self.graph, treatment)
        outcome_ancestors = nx.ancestors(self.graph, outcome)
        
        # Find common causes
        common_causes = treatment_ancestors.intersection(outcome_ancestors)
        
        # Add observed common causes to backdoor set
        for var in common_causes:
            if self.variables[var].observed:
                backdoor_set.add(var)
        
        return backdoor_set
    
    def _backdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        backdoor_set: Set[str],
        n_samples: int
    ) -> float:
        """Implement backdoor adjustment."""
        samples = self.sample(n_samples)
        
        # Stratify by backdoor variables
        effect = 0.0
        n_strata = 0
        
        # Simple implementation for binary variables
        for values in self._generate_backdoor_values(backdoor_set):
            mask = torch.ones(n_samples, dtype=torch.bool)
            for var, value in values.items():
                mask &= (samples[var] == value)
            
            if mask.sum() > 0:
                # Calculate stratum-specific effect
                treatment_outcome_corr = pearsonr(
                    samples[treatment][mask].numpy(),
                    samples[outcome][mask].numpy()
                )[0]
                
                effect += treatment_outcome_corr * (mask.sum().item() / n_samples)
                n_strata += 1
        
        return effect / n_strata if n_strata > 0 else 0.0
    
    def _generate_backdoor_values(
        self,
        backdoor_set: Set[str]
    ) -> List[Dict[str, int]]:
        """Generate all possible combinations of backdoor variable values."""
        if not backdoor_set:
            return [{}]
        
        combinations = [{}]
        for var in backdoor_set:
            var_obj = self.variables[var]
            if var_obj.type == 'binary':
                values = [0, 1]
            elif var_obj.type == 'categorical' and var_obj.categories:
                values = list(range(len(var_obj.categories)))
            else:
                # For continuous variables, discretize into bins
                values = [0, 1]  # Simplified
            
            new_combinations = []
            for combo in combinations:
                for value in values:
                    new_combo = combo.copy()
                    new_combo[var] = value
                    new_combinations.append(new_combo)
            combinations = new_combinations
        
        return combinations
    
    def compute_mutual_information(
        self,
        var1: str,
        var2: str,
        n_samples: int = 1000
    ) -> float:
        """Compute mutual information between two variables."""
        samples = self.sample(n_samples)
        
        # Convert to numpy arrays
        x = samples[var1].numpy()
        y = samples[var2].numpy()
        
        # Discretize continuous variables
        if self.variables[var1].type == 'continuous':
            x = np.digitize(x, bins=np.linspace(x.min(), x.max(), 20))
        if self.variables[var2].type == 'continuous':
            y = np.digitize(y, bins=np.linspace(y.min(), y.max(), 20))
        
        return mutual_info_score(x, y)
    
    def save_model(self, filepath: str):
        """Save the causal model structure."""
        model_data = {
            "variables": {
                name: {
                    "type": var.type,
                    "parents": list(var.parents),
                    "children": list(var.children),
                    "observed": var.observed,
                    "domain": var.domain,
                    "categories": var.categories
                }
                for name, var in self.variables.items()
            },
            "edges": list(self.graph.edges())
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'StructuralCausalModel':
        """Load a causal model structure."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        model = cls()
        
        # Add variables
        for name, var_data in model_data["variables"].items():
            model.add_variable(
                name=name,
                var_type=var_data["type"],
                parents=set(var_data["parents"]),
                observed=var_data["observed"],
                domain=var_data["domain"],
                categories=var_data["categories"]
            )
        
        return model
