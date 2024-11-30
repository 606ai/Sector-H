from typing import Dict, List
from scenarios import Scenario
from advanced_agent import AdvancedAgent
import random
import math

class CompetitionScenario(Scenario):
    def __init__(self):
        super().__init__(
            name="Resource Competition",
            description="Multiple factions compete for limited resources and territory"
        )
        self.factions = {}
        self.territories = {}
        self.resources = {}
        self.alliances = {}
        self.conflicts = []
        
    def initialize(self) -> Dict:
        # Create factions
        faction_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
        faction_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
        
        for i, name in enumerate(faction_names):
            self.factions[name] = {
                'color': faction_colors[i],
                'resources': {'energy': 100, 'materials': 100, 'influence': 50},
                'territory_count': 0,
                'agents': []
            }
            
            # Create agents for each faction
            roles = ['leader', 'warrior', 'diplomat', 'builder']
            for j, role in enumerate(roles):
                agent = AdvancedAgent(
                    name=f"{name}_{role}",
                    role=role,
                    personality={
                        'aggression': random.random(),
                        'diplomacy': random.random(),
                        'loyalty': random.uniform(0.7, 1.0)
                    }
                )
                agent.faction = name
                self.factions[name]['agents'].append(agent)
                self.agents.append(agent)
        
        # Initialize territories
        self._generate_territories(20)
        
        # Initialize resources
        self._generate_resources(30)
        
        return self.get_state()
    
    def _generate_territories(self, count: int):
        map_size = 50
        for _ in range(count):
            x = random.randint(0, map_size)
            y = random.randint(0, map_size)
            territory_id = f"{x}_{y}"
            
            self.territories[territory_id] = {
                'position': {'x': x, 'y': y},
                'controller': None,
                'value': random.uniform(0.5, 1.0),
                'defense_bonus': random.uniform(0.1, 0.3)
            }
    
    def _generate_resources(self, count: int):
        resource_types = ['energy_source', 'material_deposit', 'artifact']
        map_size = 50
        
        for _ in range(count):
            x = random.randint(0, map_size)
            y = random.randint(0, map_size)
            resource_id = f"{x}_{y}"
            
            self.resources[resource_id] = {
                'type': random.choice(resource_types),
                'position': {'x': x, 'y': y},
                'value': random.uniform(10, 100),
                'depletion_rate': random.uniform(0.1, 0.3),
                'controller': None
            }
    
    def update(self) -> Dict:
        self.time += 1
        updates = []
        
        # Update each faction's resources
        self._update_faction_resources()
        
        # Process agent actions
        for agent in self.agents:
            context = self._get_agent_context(agent)
            action = agent.think(context)
            result = self._process_action(agent, action)
            updates.append(result)
        
        # Process diplomatic relations
        self._update_diplomacy()
        
        # Generate random events
        if random.random() < 0.2:  # 20% chance per tick
            event = self._generate_competition_event()
            self._process_event(event)
            updates.append(event)
        
        return {
            'time': self.time,
            'updates': updates,
            'state': self.get_state()
        }
    
    def _update_faction_resources(self):
        for faction_name, faction in self.factions.items():
            # Base resource generation from controlled territories
            territory_count = sum(1 for t in self.territories.values() 
                                if t['controller'] == faction_name)
            
            faction['resources']['influence'] += territory_count * 0.5
            
            # Resource generation from controlled resource points
            for resource in self.resources.values():
                if resource['controller'] == faction_name:
                    if resource['type'] == 'energy_source':
                        faction['resources']['energy'] += 2 * (1 - resource['depletion_rate'])
                    elif resource['type'] == 'material_deposit':
                        faction['resources']['materials'] += 2 * (1 - resource['depletion_rate'])
    
    def _get_agent_context(self, agent: AdvancedAgent) -> Dict:
        faction = self.factions[agent.faction]
        nearby_resources = self._get_nearby_entities(agent.location, self.resources)
        nearby_territories = self._get_nearby_entities(agent.location, self.territories)
        
        return {
            'faction_resources': faction['resources'],
            'nearby_resources': nearby_resources,
            'nearby_territories': nearby_territories,
            'nearby_agents': [
                {
                    'id': a.name,
                    'faction': a.faction,
                    'role': a.role,
                    'distance': self._calculate_distance(agent.location, a.location)
                }
                for a in self.agents if a != agent
            ],
            'alliances': self.alliances.get(agent.faction, []),
            'recent_conflicts': self.conflicts[-5:]
        }
    
    def _process_action(self, agent: AdvancedAgent, action: Dict) -> Dict:
        result = {'agent': agent.name, 'faction': agent.faction, 'action': action['type'], 'success': True}
        
        if action['type'] == 'claim_territory':
            territory_id = action.get('target')
            if territory_id in self.territories:
                territory = self.territories[territory_id]
                if not territory['controller']:
                    territory['controller'] = agent.faction
                    self.factions[agent.faction]['territory_count'] += 1
                    result['claimed_territory'] = territory_id
                else:
                    result['success'] = False
        
        elif action['type'] == 'propose_alliance':
            target_faction = action.get('target')
            if (target_faction in self.factions and 
                target_faction not in self.alliances.get(agent.faction, [])):
                if random.random() < agent.personality['diplomacy']:
                    self._create_alliance(agent.faction, target_faction)
                    result['alliance_formed'] = target_faction
                else:
                    result['success'] = False
        
        return result
    
    def _create_alliance(self, faction1: str, faction2: str):
        if faction1 not in self.alliances:
            self.alliances[faction1] = []
        if faction2 not in self.alliances:
            self.alliances[faction2] = []
            
        self.alliances[faction1].append(faction2)
        self.alliances[faction2].append(faction1)
    
    def _update_diplomacy(self):
        # Decay or strengthen alliances based on actions
        for faction, allies in list(self.alliances.items()):
            for ally in allies:
                if random.random() < 0.1:  # 10% chance to test alliance
                    if random.random() > 0.7:  # 30% chance to break
                        self.alliances[faction].remove(ally)
                        self.alliances[ally].remove(faction)
    
    def _generate_competition_event(self) -> Dict:
        events = [
            {
                'type': 'resource_discovery',
                'description': 'New resource deposits discovered',
                'effect': {'new_resources': random.randint(1, 3)}
            },
            {
                'type': 'natural_disaster',
                'description': 'Natural disaster affects territories',
                'effect': {'territory_damage': random.uniform(0.1, 0.3)}
            },
            {
                'type': 'technological_breakthrough',
                'description': 'Faction makes technological advancement',
                'effect': {'faction': random.choice(list(self.factions.keys())),
                          'bonus': random.uniform(0.2, 0.4)}
            }
        ]
        return random.choice(events)
    
    def _get_nearby_entities(self, location: Dict, entities: Dict, radius: float = 10.0) -> List[Dict]:
        nearby = []
        for entity_id, entity in entities.items():
            distance = self._calculate_distance(location, entity['position'])
            if distance <= radius:
                nearby.append({
                    'id': entity_id,
                    **entity,
                    'distance': distance
                })
        return nearby
    
    def _calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        return math.sqrt(dx*dx + dy*dy)
    
    def get_state(self) -> Dict:
        return {
            'name': self.name,
            'time': self.time,
            'factions': {
                name: {
                    'color': faction['color'],
                    'resources': faction['resources'],
                    'territory_count': faction['territory_count'],
                    'agents': [a.name for a in faction['agents']]
                }
                for name, faction in self.factions.items()
            },
            'territories': self.territories,
            'resources': self.resources,
            'alliances': self.alliances,
            'conflicts': self.conflicts[-10:],
            'agents': {
                agent.name: {
                    'faction': agent.faction,
                    'role': agent.role,
                    'location': agent.location,
                    'state': agent.get_state()
                }
                for agent in self.agents
            }
        }
