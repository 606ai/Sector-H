from typing import List, Dict
from agent import Agent
import json
import random

class World:
    def __init__(self):
        self.agents: List[Agent] = []
        self.environment = {
            "resources": {},
            "locations": [],
            "time": 0,
            "events": []
        }

    def add_agent(self, agent: Agent):
        """Add a new agent to the world"""
        self.agents.append(agent)
        return f"Agent {agent.name} added to the world"

    def update(self) -> Dict:
        """Update world state"""
        self.environment["time"] += 1
        updates = []

        # Process each agent's turn
        for agent in self.agents:
            context = self._get_context_for_agent(agent)
            decision = agent.think(context)
            action_result = agent.act(decision)
            updates.append(action_result)

        # Generate random events
        if random.random() < 0.1:  # 10% chance of random event
            event = self._generate_random_event()
            self.environment["events"].append(event)
            updates.append({"type": "event", "details": event})

        return {
            "time": self.environment["time"],
            "updates": updates,
            "events": self.environment["events"][-5:]  # Last 5 events
        }

    def _get_context_for_agent(self, agent: Agent) -> Dict:
        """Get relevant context for an agent"""
        return {
            "current_time": self.environment["time"],
            "nearby_agents": [a.name for a in self.agents if a != agent],
            "available_resources": self.environment["resources"],
            "recent_events": self.environment["events"][-3:],
            "location_info": self.environment["locations"]
        }

    def _generate_random_event(self) -> Dict:
        """Generate a random world event"""
        events = [
            {"type": "resource_discovery", "details": "New resources appeared"},
            {"type": "weather_change", "details": "Weather patterns shifting"},
            {"type": "technological_breakthrough", "details": "New capabilities unlocked"}
        ]
        return random.choice(events)

    def add_location(self, name: str, properties: Dict):
        """Add a new location to the world"""
        self.environment["locations"].append({
            "name": name,
            "properties": properties
        })

    def add_resource(self, name: str, quantity: int):
        """Add resources to the world"""
        self.environment["resources"][name] = quantity

    def get_state(self) -> Dict:
        """Get current world state"""
        return {
            "time": self.environment["time"],
            "agent_count": len(self.agents),
            "resources": self.environment["resources"],
            "locations": self.environment["locations"],
            "recent_events": self.environment["events"][-5:]
        }
