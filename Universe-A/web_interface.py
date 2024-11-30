from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
from threading import Lock
from model_integration import ModelManager
from world import World
from scenarios import CivilizationScenario

app = Flask(__name__)
socketio = SocketIO(app)
thread = None
thread_lock = Lock()

# Global state
world_state = {
    'simulation_running': False,
    'current_scenario': None,
    'world': None,
    'model_manager': ModelManager()
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        'running': world_state['simulation_running'],
        'scenario': world_state['current_scenario'],
        'model_info': world_state['model_manager'].get_model_info() if world_state['model_manager'] else None
    })

@app.route('/api/start', methods=['POST'])
def start_simulation():
    data = request.json
    scenario_type = data.get('scenario', 'civilization')
    model_name = data.get('model', 'llama2')
    
    world_state['model_manager'].set_model(model_name)
    world_state['current_scenario'] = scenario_type
    world_state['simulation_running'] = True
    
    if scenario_type == 'civilization':
        world_state['world'] = CivilizationScenario()
        world_state['world'].initialize()
    
    start_background_thread()
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    world_state['simulation_running'] = False
    return jsonify({'status': 'stopped'})

def background_thread():
    while world_state['simulation_running']:
        if world_state['world']:
            update = world_state['world'].update()
            socketio.emit('world_update', update)
        socketio.sleep(1)

def start_background_thread():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

@socketio.on('connect')
def handle_connect():
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    pass

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
