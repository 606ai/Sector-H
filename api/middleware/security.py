from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import time
from typing import Optional, Dict
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Security middleware configuration for FastAPI."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self._setup_cors()
        self._setup_rate_limiting()
        self._setup_security_headers()
        self._setup_jwt_auth()
    
    def _setup_cors(self):
        """Configure CORS settings."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],  # Add production domains
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_rate_limiting(self):
        """Configure rate limiting."""
        self.app.add_middleware(RateLimitMiddleware)
    
    def _setup_security_headers(self):
        """Configure security headers."""
        self.app.add_middleware(SecurityHeadersMiddleware)
    
    def _setup_jwt_auth(self):
        """Configure JWT authentication."""
        self.app.add_middleware(JWTAuthMiddleware)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limit = 100  # requests per minute
        self.window = 60  # seconds
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # Initialize or clean old requests
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip] = [req_time for req_time in self.requests[client_ip] 
                                  if now - req_time < self.window]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.rate_limit:
            raise HTTPException(status_code=429, detail="Too many requests")
        
        self.requests[client_ip].append(now)
        response = await call_next(request)
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=()"
        }
        
        for key, value in headers.items():
            response.headers[key] = value
        
        return response

class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security = HTTPBearer()
        self.secret_key = "your-secret-key"  # Move to environment variables
        self.algorithm = "HS256"
    
    def create_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public endpoints
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/auth/login"]:
            return await call_next(request)
        
        try:
            credentials: HTTPAuthorizationCredentials = await self.security(request)
            token = credentials.credentials
            payload = self.decode_token(token)
            request.state.user = payload
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        
        return await call_next(request)
