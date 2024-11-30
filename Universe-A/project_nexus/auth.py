from functools import wraps
from flask import request, jsonify, current_app
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"  # Change this in production

class User:
    def __init__(self, id, username, email, role):
        self.id = id
        self.username = username
        self.email = email
        self.role = role

def generate_token(user):
    payload = {
        'user_id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role,
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            user = User(
                id=payload['user_id'],
                username=payload['username'],
                email=payload['email'],
                role=payload['role']
            )
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
            
    return decorated

def get_current_user():
    """For development purposes, returns a default user.
    In production, this should verify the JWT token and return the actual user."""
    return User(
        id="1",
        username="dev_user",
        email="dev@example.com",
        role="admin"
    )
