from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import torch
import gym
from gym import spaces
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """Represents the state of an agent in the environment."""
    id: str
    position: np.ndarray
    orientation: float
    velocity: np.ndarray
    observations: Dict[str, Any]
    reward: float = 0.0
    done: bool = False

class MultiAgentEnvironment(gym.Env):
    """Base class for multi-agent environments."""
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        max_steps: int = 1000,
        communication_range: float = 1.0,
        observation_range: float = 2.0
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.communication_range = communication_range
        self.observation_range = observation_range
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'self': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32
            ),
            'others': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_agents - 1, state_dim),
                dtype=np.float32
            ),
            'communication': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_agents - 1, state_dim),
                dtype=np.float32
            )
        })
        
        self.agents: Dict[str, AgentState] = {}
        self.current_step = 0
        self.episode_rewards: List[float] = []
    
    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Reset the environment."""
        self.current_step = 0
        self.episode_rewards = []
        self.agents.clear()
        
        # Initialize agents
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = AgentState(
                id=agent_id,
                position=self._random_position(),
                orientation=np.random.uniform(0, 2 * np.pi),
                velocity=np.zeros(2),
                observations=self._get_agent_observation(agent_id)
            )
        
        return self._get_observations()
    
    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Any]
    ]:
        """Execute one time step."""
        self.current_step += 1
        
        # Update agent states based on actions
        for agent_id, action in actions.items():
            self._update_agent_state(agent_id, action)
        
        # Get observations, rewards, and done flags
        observations = self._get_observations()
        rewards = self._compute_rewards()
        dones = self._check_termination()
        info = self._get_info()
        
        # Update agent states with new information
        for agent_id in self.agents:
            self.agents[agent_id].observations = observations[agent_id]
            self.agents[agent_id].reward = rewards[agent_id]
            self.agents[agent_id].done = dones[agent_id]
        
        # Track episode rewards
        self.episode_rewards.append(sum(rewards.values()))
        
        return observations, rewards, dones, info
    
    def _random_position(self) -> np.ndarray:
        """Generate random position for agent initialization."""
        return np.random.uniform(-1, 1, size=2)
    
    def _get_agent_observation(
        self,
        agent_id: str
    ) -> Dict[str, np.ndarray]:
        """Get observation for a specific agent."""
        agent = self.agents[agent_id]
        other_agents = [
            other for other in self.agents.values()
            if other.id != agent_id
        ]
        
        # Get observations of other agents within range
        visible_agents = []
        communicating_agents = []
        
        for other in other_agents:
            distance = np.linalg.norm(
                agent.position - other.position
            )
            
            if distance <= self.observation_range:
                visible_agents.append(self._get_agent_state_vector(other))
            
            if distance <= self.communication_range:
                communicating_agents.append(self._get_agent_state_vector(other))
        
        # Pad observations if necessary
        while len(visible_agents) < self.n_agents - 1:
            visible_agents.append(np.zeros(self.state_dim))
        while len(communicating_agents) < self.n_agents - 1:
            communicating_agents.append(np.zeros(self.state_dim))
        
        return {
            'self': self._get_agent_state_vector(agent),
            'others': np.array(visible_agents),
            'communication': np.array(communicating_agents)
        }
    
    def _get_agent_state_vector(
        self,
        agent: AgentState
    ) -> np.ndarray:
        """Convert agent state to vector representation."""
        return np.concatenate([
            agent.position,
            [agent.orientation],
            agent.velocity
        ])
    
    def _update_agent_state(
        self,
        agent_id: str,
        action: np.ndarray
    ):
        """Update agent state based on action."""
        agent = self.agents[agent_id]
        
        # Update velocity and position
        acceleration = action[:2]
        angular_velocity = action[2]
        
        agent.velocity += acceleration * 0.1  # Time step
        agent.velocity = np.clip(agent.velocity, -1.0, 1.0)
        
        agent.position += agent.velocity * 0.1
        agent.position = np.clip(agent.position, -1.0, 1.0)
        
        agent.orientation += angular_velocity * 0.1
        agent.orientation = agent.orientation % (2 * np.pi)
    
    def _get_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get observations for all agents."""
        return {
            agent_id: self._get_agent_observation(agent_id)
            for agent_id in self.agents
        }
    
    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards for all agents."""
        rewards = {}
        
        for agent_id, agent in self.agents.items():
            # Example reward function
            # 1. Distance to goal
            distance_reward = -np.linalg.norm(agent.position)
            
            # 2. Collision penalty
            collision_penalty = 0.0
            for other in self.agents.values():
                if other.id != agent_id:
                    distance = np.linalg.norm(
                        agent.position - other.position
                    )
                    if distance < 0.1:
                        collision_penalty -= 1.0
            
            # 3. Velocity penalty
            velocity_penalty = -0.1 * np.linalg.norm(agent.velocity)
            
            rewards[agent_id] = (
                distance_reward +
                collision_penalty +
                velocity_penalty
            )
        
        return rewards
    
    def _check_termination(self) -> Dict[str, bool]:
        """Check termination conditions."""
        dones = {}
        
        # Global termination condition
        timeout = self.current_step >= self.max_steps
        
        for agent_id, agent in self.agents.items():
            # Agent-specific termination conditions
            reached_goal = np.linalg.norm(agent.position) < 0.1
            
            # Check collisions
            collision = False
            for other in self.agents.values():
                if other.id != agent_id:
                    distance = np.linalg.norm(
                        agent.position - other.position
                    )
                    if distance < 0.1:
                        collision = True
                        break
            
            dones[agent_id] = timeout or reached_goal or collision
        
        return dones
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment."""
        return {
            'current_step': self.current_step,
            'episode_reward': sum(self.episode_rewards),
            'agent_positions': {
                agent_id: agent.position.copy()
                for agent_id, agent in self.agents.items()
            },
            'agent_velocities': {
                agent_id: agent.velocity.copy()
                for agent_id, agent in self.agents.items()
            }
        }
    
    def render(self, mode='human'):
        """Render the environment."""
        pass  # Implement visualization if needed
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
