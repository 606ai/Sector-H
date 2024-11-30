from typing import List, Dict, Optional
import random
import math
from dataclasses import dataclass
import requests
from transformers import pipeline

@dataclass
class EmotionalState:
    happiness: float = 0.5    # 0-1 scale
    stress: float = 0.5      # 0-1 scale
    curiosity: float = 0.5   # 0-1 scale
    social: float = 0.5      # 0-1 scale
    
    def update(self, events: List[Dict]):
        for event in events:
            if event['type'] == 'positive':
                self.happiness = min(1.0, self.happiness + 0.1)
                self.stress = max(0.0, self.stress - 0.05)
            elif event['type'] == 'negative':
                self.happiness = max(0.0, self.happiness - 0.1)
                self.stress = min(1.0, self.stress + 0.1)
            elif event['type'] == 'discovery':
                self.curiosity = min(1.0, self.curiosity + 0.15)
            elif event['type'] == 'social':
                self.social = min(1.0, self.social + 0.1)

class AdvancedAgent:
    def __init__(self, name: str, role: str, personality: Dict[str, float]):
        self.name = name
        self.role = role
        self.personality = personality
        self.emotional_state = EmotionalState()
        self.memory = []
        self.knowledge_base = set()
        self.relationships = {}
        self.goals = []
        self.location = {'x': 0, 'y': 0}
        self.inventory = {}
        self.skills = {}
        self.experience = 0
        
        # Initialize NLP pipeline for understanding
        try:
            self.nlp = pipeline("text-classification", model="distilbert-base-uncased")
        except Exception:
            self.nlp = None

    def think(self, context: Dict) -> Dict:
        """Enhanced decision-making process"""
        # Analyze current state
        state_assessment = self._assess_situation(context)
        
        # Generate possible actions
        possible_actions = self._generate_actions(state_assessment)
        
        # Evaluate actions based on personality and emotional state
        scored_actions = self._evaluate_actions(possible_actions)
        
        # Choose best action
        chosen_action = self._choose_action(scored_actions)
        
        # Update emotional state based on decision
        self._update_emotional_state(chosen_action)
        
        return chosen_action

    def _assess_situation(self, context: Dict) -> Dict:
        """Analyze current situation"""
        assessment = {
            'threats': [],
            'opportunities': [],
            'social_interactions': [],
            'resource_needs': []
        }
        
        # Check for immediate threats
        for event in context.get('events', []):
            if event.get('danger_level', 0) > 0.7:
                assessment['threats'].append(event)
        
        # Identify opportunities
        for resource in context.get('resources', []):
            if resource['value'] > 0.6 and resource['distance'] < 5:
                assessment['opportunities'].append(resource)
        
        # Assess social situation
        for agent in context.get('nearby_agents', []):
            relationship = self.relationships.get(agent['id'], 0)
            assessment['social_interactions'].append({
                'agent': agent['id'],
                'relationship': relationship,
                'distance': agent['distance']
            })
        
        return assessment

    def _generate_actions(self, assessment: Dict) -> List[Dict]:
        """Generate possible actions based on assessment"""
        actions = []
        
        # Handle threats
        if assessment['threats']:
            actions.append({
                'type': 'flee',
                'priority': 0.9,
                'target': assessment['threats'][0]
            })
        
        # Handle opportunities
        for opportunity in assessment['opportunities']:
            actions.append({
                'type': 'gather',
                'priority': 0.6,
                'target': opportunity
            })
        
        # Handle social interactions
        for interaction in assessment['social_interactions']:
            if interaction['relationship'] > 0.5:
                actions.append({
                    'type': 'collaborate',
                    'priority': 0.7,
                    'target': interaction['agent']
                })
        
        return actions

    def _evaluate_actions(self, actions: List[Dict]) -> List[Dict]:
        """Score actions based on personality and emotional state"""
        scored_actions = []
        
        for action in actions:
            score = action['priority']
            
            # Modify score based on personality
            if action['type'] == 'collaborate':
                score *= (0.5 + self.personality.get('extroversion', 0.5))
            elif action['type'] == 'flee':
                score *= (0.5 + self.personality.get('cautiousness', 0.5))
            
            # Modify score based on emotional state
            if action['type'] == 'collaborate':
                score *= (0.5 + self.emotional_state.social)
            elif action['type'] == 'gather':
                score *= (0.5 + self.emotional_state.curiosity)
            
            scored_actions.append({
                **action,
                'score': score
            })
        
        return sorted(scored_actions, key=lambda x: x['score'], reverse=True)

    def _choose_action(self, scored_actions: List[Dict]) -> Dict:
        """Choose the best action with some randomness"""
        if not scored_actions:
            return {'type': 'idle', 'reason': 'No valid actions available'}
        
        # Sometimes choose random action based on curiosity
        if random.random() < self.emotional_state.curiosity * 0.2:
            return random.choice(scored_actions)
        
        return scored_actions[0]

    def _update_emotional_state(self, chosen_action: Dict):
        """Update emotional state based on chosen action"""
        events = []
        
        if chosen_action['type'] == 'flee':
            events.append({'type': 'negative'})
        elif chosen_action['type'] == 'gather':
            events.append({'type': 'discovery'})
        elif chosen_action['type'] == 'collaborate':
            events.append({'type': 'social'})
        
        self.emotional_state.update(events)

    def learn(self, information: str):
        """Learn new information"""
        self.knowledge_base.add(information)
        self.experience += 1
        
        # Update skills based on experience
        skill_type = self._categorize_information(information)
        if skill_type:
            self.skills[skill_type] = self.skills.get(skill_type, 0) + 0.1

    def _categorize_information(self, information: str) -> Optional[str]:
        """Categorize new information using NLP"""
        if self.nlp:
            try:
                result = self.nlp(information)[0]
                return result['label']
            except Exception:
                pass
        return None

    def get_state(self) -> Dict:
        """Get current agent state"""
        return {
            'name': self.name,
            'role': self.role,
            'location': self.location,
            'emotional_state': vars(self.emotional_state),
            'inventory': self.inventory,
            'skills': self.skills,
            'experience': self.experience,
            'relationships': self.relationships
        }
