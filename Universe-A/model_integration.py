import requests
from typing import Dict, Optional
from transformers import pipeline
import json
import os

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.model_type = None
        self.available_models = {
            'llama2': {'type': 'ollama', 'endpoint': 'http://localhost:11434/api/generate'},
            'mistral': {'type': 'ollama', 'endpoint': 'http://localhost:11434/api/generate'},
            'codellama': {'type': 'ollama', 'endpoint': 'http://localhost:11434/api/generate'},
            'gpt2': {'type': 'huggingface', 'model': 'gpt2'},
            'bert': {'type': 'huggingface', 'model': 'bert-base-uncased'}
        }
        
        # Initialize Hugging Face pipelines
        self.hf_pipelines = {}
        
    def set_model(self, model_name: str) -> bool:
        """Set the current model to use"""
        if model_name not in self.available_models:
            return False
            
        self.model_type = self.available_models[model_name]['type']
        self.current_model = model_name
        
        # Initialize Hugging Face pipeline if needed
        if self.model_type == 'huggingface' and model_name not in self.hf_pipelines:
            try:
                self.hf_pipelines[model_name] = pipeline(
                    'text-generation',
                    model=self.available_models[model_name]['model']
                )
            except Exception as e:
                print(f"Error loading Hugging Face model: {e}")
                return False
                
        return True

    def generate_response(self, prompt: str, max_length: int = 100) -> Optional[str]:
        """Generate response using the current model"""
        if not self.current_model:
            return None
            
        try:
            if self.model_type == 'ollama':
                return self._query_ollama(prompt)
            elif self.model_type == 'huggingface':
                return self._query_huggingface(prompt, max_length)
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API"""
        try:
            response = requests.post(
                self.available_models[self.current_model]['endpoint'],
                json={
                    "model": self.current_model,
                    "prompt": prompt
                }
            )
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return ""

    def _query_huggingface(self, prompt: str, max_length: int) -> str:
        """Query Hugging Face model"""
        try:
            pipeline = self.hf_pipelines.get(self.current_model)
            if not pipeline:
                return ""
                
            response = pipeline(prompt, max_length=max_length, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            print(f"Error querying Hugging Face model: {e}")
            return ""

    def get_model_info(self) -> Dict:
        """Get information about current model"""
        return {
            'name': self.current_model,
            'type': self.model_type,
            'available_models': list(self.available_models.keys())
        }

class ResponseProcessor:
    def __init__(self):
        self.sentiment_analyzer = None
        try:
            self.sentiment_analyzer = pipeline('sentiment-analysis')
        except Exception:
            pass

    def process_response(self, response: str) -> Dict:
        """Process and analyze model response"""
        result = {
            'text': response,
            'length': len(response),
            'sentiment': None,
            'keywords': self._extract_keywords(response)
        }
        
        # Add sentiment if analyzer is available
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(response)
                result['sentiment'] = sentiment[0]
            except Exception:
                pass
                
        return result

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> list:
        """Extract key words from text"""
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        word_freq = {}
        
        # Common words to filter out
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in keywords[:max_keywords]]
