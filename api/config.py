from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Hugging Face
    huggingface_api_key: str
    
    # OpenAI (Optional)
    openai_api_key: Optional[str] = None
    
    # AWS Credentials (Optional)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    
    # Database
    database_url: str
    database_password: str
    
    # Model Configuration
    model_cache_dir: str = "./model_cache"
    default_model: str = "facebook/opt-350m"
    
    # Security
    jwt_secret: str
    api_key: str
    
    # Service Configuration
    enable_cuda: bool = True
    max_batch_size: int = 32
    inference_timeout: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Create model cache directory if it doesn't exist
Path(get_settings().model_cache_dir).mkdir(parents=True, exist_ok=True)
