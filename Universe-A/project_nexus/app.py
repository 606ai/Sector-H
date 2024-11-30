from flask import Flask, render_template
from flask_socketio import SocketIO
from api.routes import api
from api.events import register_handlers

app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this in production
socketio = SocketIO(app, cors_allowed_origins="*")

# Register API routes
app.register_blueprint(api, url_prefix='/api')

# Register WebSocket event handlers
register_handlers(socketio)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    socketio.run(app, debug=True, port=5000, host='0.0.0.0', allow_unsafe_werkzeug=True)
