from flask import Blueprint, request, jsonify
from flask_socketio import emit
from datetime import datetime
import uuid
from auth import get_current_user, requires_auth
from project_manager import ProjectManager
from collaboration import CollaborationHub

api = Blueprint('api', __name__)
project_manager = ProjectManager()
hub = CollaborationHub()

@api.route('/tasks', methods=['GET'])
@requires_auth
def get_tasks():
    user = get_current_user()
    tasks = project_manager.get_tasks(user)
    return jsonify(tasks)

@api.route('/tasks', methods=['POST'])
@requires_auth
def create_task():
    user = get_current_user()
    task_data = request.get_json()
    task = project_manager.create_task(user, task_data)
    emit('task_update', task, broadcast=True)
    return jsonify(task), 201

@api.route('/tasks/<task_id>/status', methods=['PUT'])
@requires_auth
def update_task_status(task_id):
    user = get_current_user()
    task_data = request.get_json()
    task = project_manager.update_task_status(user, task_id, task_data)
    emit('task_update', task, broadcast=True)
    return jsonify(task)

@api.route('/tasks/<task_id>', methods=['PUT'])
@requires_auth
def update_task(task_id):
    user = get_current_user()
    task_data = request.get_json()
    task = project_manager.update_task(user, task_id, task_data)
    emit('task_update', task, broadcast=True)
    return jsonify(task)

@api.route('/channels', methods=['GET'])
@requires_auth
def get_channels():
    user = get_current_user()
    channels = hub.get_channels(user)
    return jsonify(channels)

@api.route('/channels', methods=['POST'])
@requires_auth
def create_channel():
    user = get_current_user()
    channel_data = request.get_json()
    channel = hub.create_channel(user, channel_data)
    emit('channel_update', channel, broadcast=True)
    return jsonify(channel), 201

@api.route('/channels/<channel_id>/messages', methods=['GET'])
@requires_auth
def get_messages(channel_id):
    user = get_current_user()
    messages = hub.get_messages(user, channel_id)
    return jsonify(messages)

@api.route('/channels/<channel_id>/messages', methods=['POST'])
@requires_auth
def send_message(channel_id):
    user = get_current_user()
    message_data = request.get_json()
    message = hub.send_message(user, channel_id, message_data)
    emit('new_message', message, broadcast=True)
    return jsonify(message), 201

@api.route('/documents', methods=['GET'])
@requires_auth
def get_documents():
    user = get_current_user()
    documents = project_manager.get_documents(user)
    return jsonify(documents)

@api.route('/documents', methods=['POST'])
@requires_auth
def create_document():
    user = get_current_user()
    doc_data = request.get_json()
    document = project_manager.create_document(user, doc_data)
    emit('document_update', document, broadcast=True)
    return jsonify(document), 201

@api.route('/documents/<doc_id>', methods=['PUT'])
@requires_auth
def update_document(doc_id):
    user = get_current_user()
    doc_data = request.get_json()
    document = project_manager.update_document(user, doc_id, doc_data)
    emit('document_update', document, broadcast=True)
    return jsonify(document)

@api.route('/documents/<doc_id>/comments', methods=['POST'])
@requires_auth
def add_document_comment(doc_id):
    user = get_current_user()
    comment_data = request.get_json()
    comment = hub.add_document_comment(user, doc_id, comment_data)
    emit('document_comment', comment, broadcast=True)
    return jsonify(comment), 201

@api.route('/metrics', methods=['GET'])
@requires_auth
def get_metrics():
    user = get_current_user()
    metrics = project_manager.get_metrics(user)
    collab_metrics = hub.get_channel_metrics()
    metrics['collaborationMetrics'] = collab_metrics
    return jsonify(metrics)

@api.route('/search', methods=['GET'])
@requires_auth
def search():
    user = get_current_user()
    query = request.args.get('q', '')
    results = {
        'tasks': project_manager.search_tasks(user, query),
        'messages': hub.search_messages(user, query),
        'documents': hub.search_documents(user, query)
    }
    return jsonify(results)

# WebSocket event handlers
def handle_connect():
    user = get_current_user()
    if user:
        hub.user_connected(user.id)
        emit('user_online', {'user_id': user.id}, broadcast=True)

def handle_disconnect():
    user = get_current_user()
    if user:
        hub.user_disconnected(user.id)
        emit('user_offline', {'user_id': user.id}, broadcast=True)

def handle_typing(data):
    user = get_current_user()
    if user and 'channel_id' in data:
        emit('user_typing', {
            'user_id': user.id,
            'channel_id': data['channel_id']
        }, broadcast=True)
