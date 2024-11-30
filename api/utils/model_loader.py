from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional
import torch
from ..config import get_settings

class ModelLoader:
    _instance = None
    _models: Dict[str, AutoModel] = {}
    _tokenizers: Dict[str, AutoTokenizer] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() and self.settings.enable_cuda else "cpu"
    
    def load_model(self, model_name: Optional[str] = None) -> tuple[AutoModel, AutoTokenizer]:
        """Load a model and its tokenizer."""
        model_name = model_name or self.settings.default_model
        
        if model_name not in self._models:
            # Set Hugging Face token for private models
            if self.settings.huggingface_api_key:
                AutoModel.from_pretrained(
                    model_name,
                    use_auth_token=self.settings.huggingface_api_key,
                    cache_dir=self.settings.model_cache_dir
                )
                AutoTokenizer.from_pretrained(
                    model_name,
                    use_auth_token=self.settings.huggingface_api_key,
                    cache_dir=self.settings.model_cache_dir
                )
            
            # Load model and tokenizer
            self._models[model_name] = AutoModel.from_pretrained(model_name).to(self.device)
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        
        return self._models[model_name], self._tokenizers[model_name]
    
    def unload_model(self, model_name: str):
        """Unload a model and its tokenizer from memory."""
        if model_name in self._models:
            del self._models[model_name]
            del self._tokenizers[model_name]
            torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
