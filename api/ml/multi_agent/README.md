# 🤖 Multi-Agent Systems

Implementation of multi-agent environments and learning algorithms.

## 📚 Contents

- `environment.py`: Core environment implementation
- `models/`: Agent model implementations
- `utils/`: Helper functions and utilities
- `config.py`: Configuration settings

## 🚀 Quick Start

```python
from api.ml.multi_agent import Environment
from api.ml.multi_agent.models import BaseAgent

# Create environment
env = Environment(num_agents=3)

# Create agents
agents = [BaseAgent() for _ in range(3)]

# Run simulation
obs = env.reset()
for _ in range(100):
    actions = [agent.act(obs[i]) for i, agent in enumerate(agents)]
    obs, rewards, done, info = env.step(actions)
    if done:
        break
```

## 📖 Documentation

### Environment Class
The main environment class that handles agent interactions.

```python
class Environment:
    def __init__(self, num_agents: int, max_steps: int = 100):
        """
        Args:
            num_agents: Number of agents in environment
            max_steps: Maximum steps per episode
        """
        pass

    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial observations."""
        pass

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """Execute one step of the environment."""
        pass
```

### Available Models

- `BaseAgent`: Simple reactive agent
- `PPOAgent`: Agent using Proximal Policy Optimization
- `DQNAgent`: Deep Q-Network agent

## 🔧 Configuration

Key configuration options in `config.py`:

```python
ENVIRONMENT_CONFIG = {
    'max_steps': 100,
    'observation_space': (84, 84, 3),
    'action_space': 4,
}

TRAINING_CONFIG = {
    'num_episodes': 1000,
    'learning_rate': 0.001,
    'gamma': 0.99,
}
```

## 🧪 Testing

Run module tests:
```bash
pytest api/ml/multi_agent/tests/
```
