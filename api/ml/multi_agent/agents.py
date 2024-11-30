from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Represents a single experience tuple."""
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: Dict[str, torch.Tensor]
    done: bool

class Memory:
    """Experience replay memory."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Experience] = []
        self.position = 0
    
    def push(self, experience: Experience):
        """Save an experience."""
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int
    ) -> List[Experience]:
        """Sample a batch of experiences."""
        return np.random.choice(
            self.memory,
            batch_size,
            replace=False
        ).tolist()
    
    def __len__(self) -> int:
        return len(self.memory)

class AttentionModule(nn.Module):
    """Multi-head attention for agent communication."""
    
    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        key_dim: int = 64
    ):
        super().__init__()
        self.n_heads = n_heads
        self.key_dim = key_dim
        
        self.q_linear = nn.Linear(input_dim, n_heads * key_dim)
        self.k_linear = nn.Linear(input_dim, n_heads * key_dim)
        self.v_linear = nn.Linear(input_dim, n_heads * key_dim)
        self.output_linear = nn.Linear(n_heads * key_dim, input_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Multi-head attention forward pass."""
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_linear(query).view(
            batch_size, -1, self.n_heads, self.key_dim
        ).transpose(1, 2)
        k = self.k_linear(key).view(
            batch_size, -1, self.n_heads, self.key_dim
        ).transpose(1, 2)
        v = self.v_linear(value).view(
            batch_size, -1, self.n_heads, self.key_dim
        ).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.key_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.key_dim
        )
        
        return self.output_linear(output)

class ActorCritic(nn.Module):
    """Actor-Critic network with attention-based communication."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 4
    ):
        super().__init__()
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention module for communication
        self.attention = AttentionModule(
            input_dim=hidden_dim,
            n_heads=n_heads
        )
        
        # Actor network
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.actor_log_std = nn.Parameter(
            torch.zeros(1, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        # Encode own state
        own_state = self.encoder(state['self'])
        
        # Encode other agents' states
        other_states = self.encoder(
            state['others'].view(-1, state['others'].size(-1))
        ).view(state['others'].size(0), -1, -1)
        
        # Apply attention for communication
        communication = self.attention(
            own_state.unsqueeze(1),
            other_states,
            other_states
        ).squeeze(1)
        
        # Combine own state and communication
        combined = own_state + communication
        
        # Actor output
        action_mean = self.actor_mean(combined)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Sample action
        if deterministic:
            action = action_mean
        else:
            normal = Normal(action_mean, action_std)
            action = normal.rsample()
        
        # Critic value
        value = self.critic(combined)
        
        return action, action_mean, value

class MultiAgent:
    """Multi-agent reinforcement learning agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        memory_size: int = 1000000
    ):
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        self.target_actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # Copy parameters to target network
        for target_param, param in zip(
            self.target_actor_critic.parameters(),
            self.actor_critic.parameters()
        ):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_critic.actor_mean.parameters(),
            lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.actor_critic.critic.parameters(),
            lr=lr_critic
        )
        
        # Experience replay memory
        self.memory = Memory(memory_size)
    
    def select_action(
        self,
        state: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """Select action based on current state."""
        with torch.no_grad():
            state_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0)
                for k, v in state.items()
            }
            
            action, _, _ = self.actor_critic(
                state_tensor,
                deterministic
            )
            
            return action.squeeze(0).numpy()
    
    def update(
        self,
        batch_size: int
    ) -> Dict[str, float]:
        """Update agent's networks."""
        if len(self.memory) < batch_size:
            return {}
        
        # Sample batch
        experiences = self.memory.sample(batch_size)
        
        # Prepare batch
        batch_state = {
            k: torch.FloatTensor([exp.state[k] for exp in experiences])
            for k in experiences[0].state.keys()
        }
        batch_action = torch.FloatTensor(
            [exp.action for exp in experiences]
        )
        batch_reward = torch.FloatTensor(
            [exp.reward for exp in experiences]
        )
        batch_next_state = {
            k: torch.FloatTensor([exp.next_state[k] for exp in experiences])
            for k in experiences[0].next_state.keys()
        }
        batch_done = torch.FloatTensor(
            [exp.done for exp in experiences]
        )
        
        # Compute target value
        with torch.no_grad():
            _, _, next_value = self.target_actor_critic(batch_next_state)
            target_value = batch_reward + (
                1 - batch_done
            ) * self.gamma * next_value.squeeze(-1)
        
        # Update critic
        _, _, value = self.actor_critic(batch_state)
        value = value.squeeze(-1)
        critic_loss = F.mse_loss(value, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action, action_mean, value = self.actor_critic(batch_state)
        actor_loss = -value.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target network
        for target_param, param in zip(
            self.target_actor_critic.parameters(),
            self.actor_critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "value": value.mean().item()
        }
    
    def save(self, path: str):
        """Save agent's networks."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'target_actor_critic_state_dict': self.target_actor_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load agent's networks."""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(
            checkpoint['actor_critic_state_dict']
        )
        self.target_actor_critic.load_state_dict(
            checkpoint['target_actor_critic_state_dict']
        )
        self.actor_optimizer.load_state_dict(
            checkpoint['actor_optimizer_state_dict']
        )
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict']
        )
