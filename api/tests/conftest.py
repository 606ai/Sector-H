import pytest
from fastapi.testclient import TestClient
from typing import Generator
import sys
import os

# Add API directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from config import get_settings

@pytest.fixture(scope="module")
def test_app() -> Generator:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="module")
def test_settings():
    """Get test settings."""
    return get_settings()
