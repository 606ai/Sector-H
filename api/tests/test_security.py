import pytest
from fastapi.testclient import TestClient
from datetime import timedelta
import time
from middleware.security import JWTAuthMiddleware, RateLimitMiddleware

def test_rate_limiting(test_app: TestClient):
    """Test rate limiting middleware."""
    # Make multiple requests
    for _ in range(100):
        response = test_app.get("/")
        assert response.status_code != 429
    
    # Next request should be rate limited
    response = test_app.get("/")
    assert response.status_code == 429
    assert response.json()["detail"] == "Too many requests"

def test_security_headers(test_app: TestClient):
    """Test security headers are properly set."""
    response = test_app.get("/")
    headers = response.headers
    
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-XSS-Protection"] == "1; mode=block"
    assert "Strict-Transport-Security" in headers
    assert "Content-Security-Policy" in headers
    assert "Referrer-Policy" in headers

def test_jwt_auth(test_app: TestClient):
    """Test JWT authentication."""
    # Test without token
    response = test_app.get("/protected")
    assert response.status_code == 401
    
    # Test with invalid token
    response = test_app.get(
        "/protected",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
    
    # Test with valid token
    middleware = JWTAuthMiddleware(None)
    token = middleware.create_token(
        {"sub": "test@example.com"},
        expires_delta=timedelta(minutes=30)
    )
    
    response = test_app.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200

def test_cors(test_app: TestClient):
    """Test CORS configuration."""
    response = test_app.options(
        "/",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "Content-Type" in response.headers["access-control-allow-headers"]
