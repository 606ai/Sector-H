from typing import Dict, List
from scenarios import Scenario
from advanced_agent import AdvancedAgent
import random
import math

class ExplorationScenario(Scenario):
    def __init__(self):
        super().__init__(
            name="Space Exploration",
            description="Agents explore vast regions of space, discovering new phenomena and resources"
        )
        self.map_size = 100
        self.discovered_regions = set()
        self.phenomena = {}
        self.space_events = []
        
    def initialize(self) -> Dict:
        # Create explorer agents
        roles = ['scout', 'scientist', 'navigator', 'engineer']
        for i in range(4):
            agent = AdvancedAgent(
                name=f"Explorer_{i}",
                role=roles[i],
                personality={
                    'curiosity': random.random() * 0.5 + 0.5,  # High curiosity
                    'cautiousness': random.random(),
                    'adaptability': random.random() * 0.5 + 0.5  # High adaptability
                }
            )
            self.agents.append(agent)
            
        # Generate initial phenomena
        self._generate_space_phenomena(10)
        
        return self.get_state()
    
    def _generate_space_phenomena(self, count: int):
        phenomena_types = [
            'nebula', 'black_hole', 'star_cluster', 'asteroid_field',
            'quantum_anomaly', 'wormhole', 'alien_artifact'
        ]
        
        for _ in range(count):
            x = random.randint(0, self.map_size)
            y = random.randint(0, self.map_size)
            phenomenon_type = random.choice(phenomena_types)
            
            self.phenomena[f"{x}_{y}"] = {
                'type': phenomenon_type,
                'position': {'x': x, 'y': y},
                'discovered': False,
                'studied': 0,  # Progress in studying the phenomenon
                'properties': self._generate_phenomenon_properties(phenomenon_type)
            }
    
    def _generate_phenomenon_properties(self, phenomenon_type: str) -> Dict:
        properties = {
            'size': random.uniform(1.0, 10.0),
            'energy_level': random.uniform(0.1, 1.0),
            'stability': random.uniform(0.3, 1.0)
        }
        
        if phenomenon_type == 'black_hole':
            properties.update({
                'mass': random.uniform(1000, 10000),
                'hawking_radiation': random.uniform(0.1, 0.5)
            })
        elif phenomenon_type == 'alien_artifact':
            properties.update({
                'age': random.uniform(1000, 1000000),
                'technology_level': random.uniform(0.1, 1.0),
                'decoded': 0.0
            })
            
        return properties
    
    def update(self) -> Dict:
        self.time += 1
        updates = []
        
        # Update each agent
        for agent in self.agents:
            context = self._get_agent_context(agent)
            action = agent.think(context)
            result = self._process_action(agent, action)
            updates.append(result)
        
        # Random space events
        if random.random() < 0.15:  # 15% chance per tick
            event = self._generate_space_event()
            self._process_event(event)
            updates.append(event)
        
        # Update phenomena
        self._update_phenomena()
        
        return {
            'time': self.time,
            'updates': updates,
            'state': self.get_state()
        }
    
    def _get_agent_context(self, agent: AdvancedAgent) -> Dict:
        nearby_phenomena = self._get_nearby_phenomena(agent.location)
        return {
            'nearby_phenomena': nearby_phenomena,
            'nearby_agents': [
                {
                    'id': a.name,
                    'role': a.role,
                    'distance': self._calculate_distance(agent.location, a.location)
                }
                for a in self.agents if a != agent
            ],
            'discovered_regions': len(self.discovered_regions),
            'recent_events': self.space_events[-5:]
        }
    
    def _get_nearby_phenomena(self, location: Dict, radius: float = 10.0) -> List[Dict]:
        nearby = []
        for pos, phenomenon in self.phenomena.items():
            distance = self._calculate_distance(location, phenomenon['position'])
            if distance <= radius:
                nearby.append({
                    'type': phenomenon['type'],
                    'distance': distance,
                    'discovered': phenomenon['discovered']
                })
        return nearby
    
    def _process_action(self, agent: AdvancedAgent, action: Dict) -> Dict:
        result = {'agent': agent.name, 'action': action['type'], 'success': True}
        
        if action['type'] == 'explore':
            # Add current position to discovered regions
            pos_key = f"{int(agent.location['x'])}_{int(agent.location['y'])}"
            self.discovered_regions.add(pos_key)
            
            # Check for phenomena discovery
            for pos, phenomenon in self.phenomena.items():
                if not phenomenon['discovered']:
                    distance = self._calculate_distance(agent.location, phenomenon['position'])
                    if distance < 2.0:  # Discovery radius
                        phenomenon['discovered'] = True
                        result['discovery'] = {
                            'type': phenomenon['type'],
                            'position': phenomenon['position']
                        }
        
        elif action['type'] == 'study':
            # Study nearby phenomenon
            target = action.get('target')
            if target:
                phenomenon = self.phenomena.get(f"{target['x']}_{target['y']}")
                if phenomenon and phenomenon['discovered']:
                    phenomenon['studied'] = min(1.0, phenomenon['studied'] + 0.1)
                    result['study_progress'] = phenomenon['studied']
        
        return result
    
    def _generate_space_event(self) -> Dict:
        events = [
            {
                'type': 'solar_flare',
                'description': 'Intense solar activity detected',
                'effect': {'energy_boost': random.uniform(0.1, 0.3)}
            },
            {
                'type': 'cosmic_storm',
                'description': 'Powerful cosmic storm approaching',
                'effect': {'movement_penalty': random.uniform(0.1, 0.3)}
            },
            {
                'type': 'quantum_fluctuation',
                'description': 'Strange quantum effects observed',
                'effect': {'discovery_bonus': random.uniform(0.2, 0.4)}
            }
        ]
        event = random.choice(events)
        self.space_events.append(event)
        return event
    
    def _update_phenomena(self):
        # Update dynamic properties of phenomena
        for phenomenon in self.phenomena.values():
            if phenomenon['type'] == 'quantum_anomaly':
                # Quantum anomalies shift positions
                if random.random() < 0.1:  # 10% chance to move
                    phenomenon['position']['x'] += random.uniform(-1, 1)
                    phenomenon['position']['y'] += random.uniform(-1, 1)
            elif phenomenon['type'] == 'wormhole':
                # Wormholes fluctuate in stability
                phenomenon['properties']['stability'] = max(0.1, min(1.0,
                    phenomenon['properties']['stability'] + random.uniform(-0.1, 0.1)))
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        return math.sqrt(dx*dx + dy*dy)
    
    def get_state(self) -> Dict:
        return {
            'name': self.name,
            'time': self.time,
            'discovered_regions': len(self.discovered_regions),
            'phenomena': {
                k: {
                    'type': v['type'],
                    'position': v['position'],
                    'discovered': v['discovered'],
                    'studied': v['studied']
                } for k, v in self.phenomena.items() if v['discovered']
            },
            'agents': {
                agent.name: {
                    'role': agent.role,
                    'location': agent.location,
                    'state': agent.get_state()
                } for agent in self.agents
            },
            'events': self.space_events[-10:]
        }
