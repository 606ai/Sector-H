from typing import Dict, List
from advanced_agent import AdvancedAgent
import random
import json

class Scenario:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.agents = []
        self.resources = {}
        self.structures = {}
        self.events = []
        self.rules = {}
        self.objectives = []
        self.time = 0

    def initialize(self) -> Dict:
        """Initialize the scenario"""
        raise NotImplementedError("Subclasses must implement initialize()")

    def update(self) -> Dict:
        """Update scenario state"""
        raise NotImplementedError("Subclasses must implement update()")

class CivilizationScenario(Scenario):
    def __init__(self):
        super().__init__(
            name="Civilization Building",
            description="Agents must work together to build and grow a civilization"
        )
        self.technology_level = 0
        self.population = 0
        self.happiness = 0.5
        self.resources = {
            'food': 100,
            'wood': 100,
            'stone': 100,
            'knowledge': 0
        }

    def initialize(self) -> Dict:
        # Create initial agents with different roles
        roles = ['builder', 'farmer', 'researcher', 'explorer']
        for i in range(4):
            agent = AdvancedAgent(
                name=f"Agent_{i}",
                role=roles[i],
                personality={
                    'extroversion': random.random(),
                    'cautiousness': random.random(),
                    'creativity': random.random()
                }
            )
            self.agents.append(agent)

        # Set initial structures
        self.structures = {
            'town_center': {'x': 20, 'y': 20, 'size': 2, 'type': 'central'},
            'farm': {'x': 18, 'y': 18, 'size': 1, 'type': 'production'}
        }

        return self.get_state()

    def update(self) -> Dict:
        self.time += 1
        updates = []

        # Resource production
        self._update_resources()

        # Agent actions
        for agent in self.agents:
            context = self._get_agent_context(agent)
            action = agent.think(context)
            result = self._process_action(agent, action)
            updates.append(result)

        # Random events
        if random.random() < 0.1:
            event = self._generate_event()
            self._process_event(event)
            updates.append(event)

        # Technology advancement
        if self.resources['knowledge'] >= 100:
            self._advance_technology()

        return {
            'time': self.time,
            'updates': updates,
            'state': self.get_state()
        }

    def _update_resources(self):
        """Update resource quantities"""
        # Basic resource production
        self.resources['food'] += sum(1 for agent in self.agents if agent.role == 'farmer')
        self.resources['wood'] += sum(1 for agent in self.agents if agent.role == 'builder') * 0.5
        self.resources['knowledge'] += sum(1 for agent in self.agents if agent.role == 'researcher') * 0.2

        # Resource consumption
        population_consumption = len(self.agents) * 0.5
        self.resources['food'] = max(0, self.resources['food'] - population_consumption)

    def _get_agent_context(self, agent: AdvancedAgent) -> Dict:
        """Get context for agent decision making"""
        return {
            'resources': self.resources,
            'nearby_agents': [
                {
                    'id': a.name,
                    'role': a.role,
                    'distance': self._calculate_distance(agent, a)
                }
                for a in self.agents if a != agent
            ],
            'structures': self.structures,
            'technology_level': self.technology_level,
            'events': self.events[-5:]  # Last 5 events
        }

    def _process_action(self, agent: AdvancedAgent, action: Dict) -> Dict:
        """Process agent action and return result"""
        result = {'agent': agent.name, 'action': action['type'], 'success': True}

        if action['type'] == 'gather':
            resource = action['target']['type']
            if resource in self.resources:
                gathered = random.uniform(0.5, 1.5)
                self.resources[resource] += gathered
                result['gathered'] = gathered

        elif action['type'] == 'collaborate':
            target_agent = next((a for a in self.agents if a.name == action['target']), None)
            if target_agent:
                # Collaboration increases efficiency
                agent.skills[target_agent.role] = agent.skills.get(target_agent.role, 0) + 0.1
                result['skill_increase'] = target_agent.role

        return result

    def _generate_event(self) -> Dict:
        """Generate random event"""
        events = [
            {
                'type': 'discovery',
                'description': 'New resource deposit found',
                'effect': {'resource': random.choice(list(self.resources.keys())), 'amount': 50}
            },
            {
                'type': 'disaster',
                'description': 'Natural disaster',
                'effect': {'resource': random.choice(list(self.resources.keys())), 'amount': -30}
            },
            {
                'type': 'visitor',
                'description': 'Wandering trader arrived',
                'effect': {'knowledge': 10}
            }
        ]
        return random.choice(events)

    def _process_event(self, event: Dict):
        """Process world event"""
        if 'effect' in event:
            if 'resource' in event['effect']:
                resource = event['effect']['resource']
                self.resources[resource] += event['effect']['amount']
                self.resources[resource] = max(0, self.resources[resource])

    def _advance_technology(self):
        """Advance civilization technology level"""
        self.technology_level += 1
        self.resources['knowledge'] -= 100
        
        # Add new structures or capabilities based on technology level
        if self.technology_level == 1:
            self.structures['library'] = {'x': 22, 'y': 22, 'size': 1, 'type': 'knowledge'}
        elif self.technology_level == 2:
            self.structures['workshop'] = {'x': 24, 'y': 24, 'size': 1, 'type': 'production'}

    def _calculate_distance(self, agent1: AdvancedAgent, agent2: AdvancedAgent) -> float:
        """Calculate distance between two agents"""
        dx = agent1.location['x'] - agent2.location['x']
        dy = agent1.location['y'] - agent2.location['y']
        return math.sqrt(dx*dx + dy*dy)

    def get_state(self) -> Dict:
        """Get current scenario state"""
        return {
            'name': self.name,
            'time': self.time,
            'technology_level': self.technology_level,
            'resources': self.resources,
            'structures': self.structures,
            'agents': {
                agent.name: {
                    'role': agent.role,
                    'location': agent.location,
                    'state': agent.get_state()
                }
                for agent in self.agents
            },
            'events': self.events[-10:]  # Last 10 events
        }
