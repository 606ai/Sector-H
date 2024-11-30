import requests
import json
from typing import List, Dict

class Agent:
    def __init__(self, name: str, role: str, goals: List[str]):
        self.name = name
        self.role = role
        self.goals = goals
        self.memory = []
        self.state = "idle"
        self.relationships = {}

    def think(self, context: Dict) -> str:
        """Process current context and decide next action"""
        # Here we'll integrate with Ollama for decision making
        prompt = self._create_prompt(context)
        response = self._query_ollama(prompt)
        return response

    def act(self, action: str) -> Dict:
        """Execute an action in the world"""
        self.memory.append(action)
        return {
            "agent": self.name,
            "action": action,
            "status": "completed"
        }

    def update_relationships(self, other_agent: str, relationship_type: str):
        """Update relationships with other agents"""
        self.relationships[other_agent] = relationship_type

    def _create_prompt(self, context: Dict) -> str:
        """Create a prompt for the AI model"""
        return f"""
        Agent: {self.name}
        Role: {self.role}
        Goals: {', '.join(self.goals)}
        Context: {json.dumps(context)}
        Memory: {self.memory[-5:] if self.memory else 'No previous actions'}
        
        Based on the above information, what should the agent do next?
        """

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API for decision making"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt
                }
            )
            return response.json().get("response", "Unable to decide action")
        except Exception as e:
            return f"Error querying Ollama: {str(e)}"
