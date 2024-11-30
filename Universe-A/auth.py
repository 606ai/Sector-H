from flask_login import LoginManager, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Optional
import jwt
import datetime
from functools import wraps
from flask import request, jsonify
import os

class User(UserMixin):
    def __init__(self, user_id: str, username: str, email: str, role: str = 'user'):
        self.id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.preferences = {}
        self.scenarios = []
        self.last_login = None

class UserManager:
    def __init__(self):
        self.users = {}
        self.login_manager = LoginManager()
        self.secret_key = os.environ.get('JWT_SECRET_KEY', 'your-secret-key')

    def init_app(self, app):
        self.login_manager.init_app(app)
        
        @self.login_manager.user_loader
        def load_user(user_id):
            return self.get_user_by_id(user_id)

    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> Optional[User]:
        if self.get_user_by_email(email):
            return None
            
        user_id = str(len(self.users) + 1)
        user = User(user_id, username, email, role)
        user.password_hash = generate_password_hash(password)
        self.users[user_id] = user
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        return next((user for user in self.users.values() if user.email == email), None)

    def verify_password(self, user: User, password: str) -> bool:
        return check_password_hash(user.password_hash, password)

    def generate_token(self, user: User) -> str:
        payload = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
            
        user_manager = UserManager()  # You might want to pass this as a parameter instead
        payload = user_manager.verify_token(token)
        
        if not payload:
            return jsonify({'message': 'Invalid token'}), 401
            
        return f(*args, **kwargs)
    
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
            
        user_manager = UserManager()
        payload = user_manager.verify_token(token)
        
        if not payload:
            return jsonify({'message': 'Invalid token'}), 401
            
        if payload.get('role') != 'admin':
            return jsonify({'message': 'Admin privileges required'}), 403
            
        return f(*args, **kwargs)
    
    return decorated_function
