from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import DataLoader
import copy
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from .search_space import SearchSpace, ArchitectureGenerator, HardwareEstimator

logger = logging.getLogger(__name__)

@dataclass
class Objective:
    """Represents an optimization objective."""
    name: str
    weight: float
    minimize: bool = True

class Individual:
    """Represents an individual architecture in the population."""
    
    def __init__(
        self,
        architecture: Dict[str, Any],
        fitness: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        self.architecture = architecture
        self.fitness = fitness
        self.metrics = metrics or {}
    
    def clone(self) -> 'Individual':
        """Create a deep copy of the individual."""
        return Individual(
            architecture=copy.deepcopy(self.architecture),
            fitness=self.fitness,
            metrics=copy.deepcopy(self.metrics)
        )

class EvolutionaryOptimizer:
    """Evolutionary algorithm for neural architecture search."""
    
    def __init__(
        self,
        search_space: SearchSpace,
        objectives: List[Objective],
        population_size: int = 50,
        tournament_size: int = 5,
        mutation_rate: float = 0.1,
        n_workers: int = 4
    ):
        self.search_space = search_space
        self.objectives = objectives
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.n_workers = n_workers
        
        self.arch_generator = ArchitectureGenerator(search_space)
        self.hardware_estimator = HardwareEstimator()
        self.population: List[Individual] = []
        self.history: List[Dict[str, Any]] = []
    
    def initialize_population(self) -> List[Individual]:
        """Initialize random population."""
        return [
            Individual(self.search_space.sample_architecture())
            for _ in range(self.population_size)
        ]
    
    def evaluate_individual(
        self,
        individual: Individual,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 1
    ) -> Dict[str, float]:
        """Evaluate an individual architecture."""
        try:
            # Generate model
            model = self.arch_generator.generate_model(individual.architecture)
            model = model.cuda()
            
            # Training setup
            optimizer = torch.optim.Adam(model.parameters())
            criterion = torch.nn.CrossEntropyLoss()
            
            # Quick training
            model.train()
            for epoch in range(n_epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.cuda(), target.cuda()
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.cuda(), target.cuda()
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            val_loss /= len(val_loader)
            accuracy = correct / total
            
            # Hardware estimates
            flops = self.hardware_estimator.estimate_flops(
                individual.architecture,
                self.search_space.input_shape
            )
            memory = self.hardware_estimator.estimate_memory(
                individual.architecture,
                self.search_space.input_shape
            )
            
            return {
                "val_loss": val_loss,
                "accuracy": accuracy,
                "flops": flops,
                "memory": memory
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate individual: {str(e)}")
            return {
                "val_loss": float('inf'),
                "accuracy": 0.0,
                "flops": float('inf'),
                "memory": float('inf')
            }
    
    def calculate_fitness(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """Calculate fitness from multiple objectives."""
        fitness = 0.0
        
        for objective in self.objectives:
            value = metrics.get(objective.name, float('inf') if objective.minimize else 0.0)
            
            # Normalize and combine objectives
            if objective.minimize:
                fitness -= objective.weight * value
            else:
                fitness += objective.weight * value
        
        return fitness
    
    def tournament_selection(
        self,
        population: List[Individual],
        tournament_size: int
    ) -> Individual:
        """Select individual using tournament selection."""
        tournament = np.random.choice(
            population,
            size=tournament_size,
            replace=False
        )
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Individual:
        """Perform crossover between two parent architectures."""
        child_arch = copy.deepcopy(parent1.architecture)
        
        # Layer-wise crossover
        if len(parent1.architecture["layers"]) > 1 and len(parent2.architecture["layers"]) > 1:
            crossover_point = np.random.randint(1, min(
                len(parent1.architecture["layers"]),
                len(parent2.architecture["layers"])
            ))
            
            child_arch["layers"] = (
                parent1.architecture["layers"][:crossover_point] +
                parent2.architecture["layers"][crossover_point:]
            )
        
        return Individual(child_arch)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual architecture."""
        mutated = individual.clone()
        
        if np.random.random() < self.mutation_rate:
            # Randomly select mutation type
            mutation_type = np.random.choice([
                "add_layer",
                "remove_layer",
                "modify_layer"
            ])
            
            if mutation_type == "add_layer" and len(mutated.architecture["layers"]) < self.search_space.max_layers:
                # Add new layer at random position
                position = np.random.randint(0, len(mutated.architecture["layers"]))
                new_layer = self.search_space._sample_layer(self.search_space.input_shape)
                mutated.architecture["layers"].insert(position, new_layer)
            
            elif mutation_type == "remove_layer" and len(mutated.architecture["layers"]) > 1:
                # Remove random layer (except classification)
                position = np.random.randint(0, len(mutated.architecture["layers"]) - 1)
                mutated.architecture["layers"].pop(position)
            
            elif mutation_type == "modify_layer":
                # Modify random layer parameters
                position = np.random.randint(0, len(mutated.architecture["layers"]) - 1)
                layer = mutated.architecture["layers"][position]
                
                if layer["type"] in [OpType.CONV.value, OpType.SEPARABLE_CONV.value]:
                    if np.random.random() < 0.5:
                        layer["filters"] = 2 ** np.random.randint(
                            4,
                            int(np.log2(self.search_space.max_filters)) + 1
                        )
                    else:
                        layer["kernel_size"] = np.random.randint(
                            1,
                            self.search_space.max_kernel_size + 1
                        )
                
                elif layer["type"] == OpType.ATTENTION.value:
                    if np.random.random() < 0.5:
                        layer["heads"] = 2 ** np.random.randint(1, 4)
                    else:
                        layer["key_dim"] = 2 ** np.random.randint(4, 7)
        
        return mutated
    
    def evolve(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_generations: int = 50,
        n_epochs: int = 1
    ) -> Tuple[Individual, List[Dict[str, Any]]]:
        """Run evolutionary optimization."""
        try:
            # Initialize population
            self.population = self.initialize_population()
            
            # Evaluate initial population
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                metrics_list = list(executor.map(
                    lambda x: self.evaluate_individual(x, train_loader, val_loader, n_epochs),
                    self.population
                ))
            
            for ind, metrics in zip(self.population, metrics_list):
                ind.metrics = metrics
                ind.fitness = self.calculate_fitness(metrics)
            
            # Evolution loop
            for generation in range(n_generations):
                new_population = []
                
                # Elitism: keep best individual
                elite = max(self.population, key=lambda x: x.fitness)
                new_population.append(elite.clone())
                
                # Generate new individuals
                while len(new_population) < self.population_size:
                    # Selection
                    parent1 = self.tournament_selection(
                        self.population,
                        self.tournament_size
                    )
                    parent2 = self.tournament_selection(
                        self.population,
                        self.tournament_size
                    )
                    
                    # Crossover
                    child = self.crossover(parent1, parent2)
                    
                    # Mutation
                    child = self.mutate(child)
                    
                    new_population.append(child)
                
                # Evaluate new population
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    metrics_list = list(executor.map(
                        lambda x: self.evaluate_individual(x, train_loader, val_loader, n_epochs),
                        new_population
                    ))
                
                for ind, metrics in zip(new_population, metrics_list):
                    ind.metrics = metrics
                    ind.fitness = self.calculate_fitness(metrics)
                
                # Update population
                self.population = new_population
                
                # Record history
                gen_stats = {
                    "generation": generation,
                    "best_fitness": elite.fitness,
                    "best_metrics": elite.metrics,
                    "avg_fitness": np.mean([ind.fitness for ind in self.population]),
                    "avg_metrics": {
                        k: np.mean([ind.metrics[k] for ind in self.population])
                        for k in self.population[0].metrics
                    }
                }
                self.history.append(gen_stats)
                
                logger.info(f"Generation {generation}: Best fitness = {elite.fitness:.4f}")
            
            return max(self.population, key=lambda x: x.fitness), self.history
            
        except Exception as e:
            logger.error(f"Evolution failed: {str(e)}")
            raise
