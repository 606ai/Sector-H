from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import torch
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
import networkx as nx
import logging
from .structural_model import StructuralCausalModel, CausalVariable

logger = logging.getLogger(__name__)

class CausalDiscovery:
    """Causal structure learning and discovery."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        variable_types: Dict[str, str],
        alpha: float = 0.05
    ):
        self.data = data
        self.variable_types = variable_types
        self.alpha = alpha
        self.graph = nx.DiGraph()
        self.skeleton = nx.Graph()
        self.separating_sets: Dict[Tuple[str, str], Set[str]] = {}
    
    def learn_structure(
        self,
        method: str = 'pc',
        max_cond_vars: int = 3
    ) -> StructuralCausalModel:
        """Learn causal structure from data."""
        if method == 'pc':
            return self._pc_algorithm(max_cond_vars)
        elif method == 'granger':
            return self._granger_causality()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _pc_algorithm(
        self,
        max_cond_vars: int
    ) -> StructuralCausalModel:
        """Implement the PC algorithm for causal discovery."""
        variables = list(self.data.columns)
        n_vars = len(variables)
        
        # Initialize complete undirected graph
        self.skeleton = nx.complete_graph(variables)
        
        # Step 1: Learn the skeleton
        for d in range(max_cond_vars + 1):
            separations = []
            for (x, y) in self.skeleton.edges():
                adj_x = set(self.skeleton.neighbors(x)) - {y}
                if len(adj_x) >= d:
                    for S in self._combinations(adj_x, d):
                        if self._conditional_independent(x, y, S):
                            separations.append((x, y))
                            self.separating_sets[(x, y)] = S
                            self.separating_sets[(y, x)] = S
                            break
            
            # Remove edges based on conditional independence
            self.skeleton.remove_edges_from(separations)
        
        # Step 2: Orient edges
        self.graph = self._orient_edges()
        
        # Create Structural Causal Model
        scm = StructuralCausalModel()
        
        # Add variables to SCM
        for var in variables:
            parents = set(self.graph.predecessors(var))
            scm.add_variable(
                name=var,
                var_type=self.variable_types[var],
                parents=parents
            )
        
        return scm
    
    def _granger_causality(self) -> StructuralCausalModel:
        """Implement Granger causality for time series data."""
        variables = list(self.data.columns)
        max_lag = 5  # Maximum lag to consider
        
        # Initialize graph
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(variables)
        
        # Test Granger causality for each pair
        for x in variables:
            for y in variables:
                if x != y:
                    # Prepare lagged data
                    X = pd.concat([
                        self.data[x].shift(i)
                        for i in range(1, max_lag + 1)
                    ], axis=1).dropna()
                    
                    y_data = self.data[y][max_lag:]
                    
                    # Fit models
                    model_with_x = LinearRegression().fit(X, y_data)
                    score_with_x = model_with_x.score(X, y_data)
                    
                    # Compare with restricted model
                    if score_with_x > 0.1:  # Simple threshold
                        self.graph.add_edge(x, y)
        
        # Create Structural Causal Model
        scm = StructuralCausalModel()
        
        # Add variables to SCM
        for var in variables:
            parents = set(self.graph.predecessors(var))
            scm.add_variable(
                name=var,
                var_type=self.variable_types[var],
                parents=parents
            )
        
        return scm
    
    def _conditional_independent(
        self,
        x: str,
        y: str,
        conditioning_set: Set[str]
    ) -> bool:
        """Test conditional independence."""
        if not conditioning_set:
            # Unconditional independence test
            if self.variable_types[x] == 'continuous' and self.variable_types[y] == 'continuous':
                return self._pearson_correlation_test(x, y)
            else:
                return self._mutual_information_test(x, y)
        
        # Conditional independence test
        data_subset = self.data[[x, y] + list(conditioning_set)]
        
        if all(self.variable_types[v] == 'continuous' for v in data_subset.columns):
            return self._partial_correlation_test(x, y, conditioning_set)
        else:
            return self._conditional_mutual_information_test(x, y, conditioning_set)
    
    def _pearson_correlation_test(
        self,
        x: str,
        y: str
    ) -> bool:
        """Perform Pearson correlation test."""
        corr, p_value = stats.pearsonr(
            self.data[x],
            self.data[y]
        )
        return p_value > self.alpha
    
    def _mutual_information_test(
        self,
        x: str,
        y: str
    ) -> bool:
        """Perform mutual information test."""
        mi = mutual_info_score(
            self.data[x],
            self.data[y]
        )
        threshold = 0.1  # Adjust based on your needs
        return mi < threshold
    
    def _partial_correlation_test(
        self,
        x: str,
        y: str,
        conditioning_set: Set[str]
    ) -> bool:
        """Perform partial correlation test."""
        # Fit linear regression for x and y on conditioning variables
        X = self.data[list(conditioning_set)]
        
        x_resid = LinearRegression().fit(X, self.data[x]).residuals_
        y_resid = LinearRegression().fit(X, self.data[y]).residuals_
        
        # Test correlation of residuals
        corr, p_value = stats.pearsonr(x_resid, y_resid)
        return p_value > self.alpha
    
    def _conditional_mutual_information_test(
        self,
        x: str,
        y: str,
        conditioning_set: Set[str]
    ) -> bool:
        """Perform conditional mutual information test."""
        # Discretize continuous variables
        data_subset = self.data[[x, y] + list(conditioning_set)].copy()
        
        for col in data_subset.columns:
            if self.variable_types[col] == 'continuous':
                data_subset[col] = pd.qcut(
                    data_subset[col],
                    q=5,
                    labels=False
                )
        
        # Calculate conditional mutual information
        cmi = self._compute_cmi(
            data_subset[x],
            data_subset[y],
            data_subset[list(conditioning_set)]
        )
        
        threshold = 0.1  # Adjust based on your needs
        return cmi < threshold
    
    def _compute_cmi(
        self,
        x: pd.Series,
        y: pd.Series,
        z: pd.DataFrame
    ) -> float:
        """Compute conditional mutual information."""
        # Simplified implementation using discretized values
        joint_counts = pd.crosstab(
            [x, y],
            [z[col] for col in z.columns]
        )
        
        cmi = 0.0
        total = len(x)
        
        for z_val in joint_counts.columns:
            # Get probabilities
            p_z = len(z[z == z_val]) / total
            if p_z > 0:
                x_y_given_z = joint_counts[z_val].unstack()
                x_y_given_z = x_y_given_z / x_y_given_z.sum()
                
                # Calculate CMI contribution
                for i in x_y_given_z.index:
                    for j in x_y_given_z.columns:
                        p_xy_z = x_y_given_z.loc[i, j]
                        if p_xy_z > 0:
                            p_x_z = x_y_given_z.loc[i, :].sum()
                            p_y_z = x_y_given_z.loc[:, j].sum()
                            cmi += p_xy_z * p_z * np.log(p_xy_z / (p_x_z * p_y_z))
        
        return cmi
    
    def _combinations(
        self,
        items: Set[str],
        r: int
    ) -> List[Set[str]]:
        """Generate all r-combinations of items."""
        if r == 0:
            return [set()]
        
        if not items:
            return []
        
        result = []
        items = list(items)
        
        for i in range(len(items)):
            item = items[i]
            remaining = items[i + 1:]
            
            for combo in self._combinations(remaining, r - 1):
                result.append({item} | combo)
        
        return result
    
    def _orient_edges(self) -> nx.DiGraph:
        """Orient edges in the skeleton graph."""
        G = nx.DiGraph()
        G.add_nodes_from(self.skeleton.nodes())
        
        # Find v-structures
        for b in self.skeleton.nodes():
            for a, c in self._combinations(set(self.skeleton.neighbors(b)), 2):
                if not self.skeleton.has_edge(a, c):
                    if b not in self.separating_sets.get((a, c), set()):
                        # Found a v-structure: a -> b <- c
                        G.add_edge(a, b)
                        G.add_edge(c, b)
        
        # Orient remaining edges using rules
        changed = True
        while changed:
            changed = False
            
            # Rule 1: Orient edge to avoid new v-structure
            for a, b in self.skeleton.edges():
                if not (G.has_edge(a, b) or G.has_edge(b, a)):
                    for c in self.skeleton.neighbors(b):
                        if G.has_edge(b, c) and not self.skeleton.has_edge(a, c):
                            G.add_edge(a, b)
                            changed = True
                            break
            
            # Rule 2: Orient edge to avoid cycle
            if not changed:
                for a, b in self.skeleton.edges():
                    if not (G.has_edge(a, b) or G.has_edge(b, a)):
                        if nx.has_path(G, b, a):
                            G.add_edge(a, b)
                            changed = True
        
        # Add remaining edges as undirected (arbitrary orientation)
        for a, b in self.skeleton.edges():
            if not (G.has_edge(a, b) or G.has_edge(b, a)):
                G.add_edge(a, b)
        
        return G
