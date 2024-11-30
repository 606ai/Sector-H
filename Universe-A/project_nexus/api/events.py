from flask_socketio import emit
from Universe_A.project_nexus.auth import get_current_user
from Universe_A.project_nexus.project_manager import ProjectManager
from Universe_A.project_nexus.collaboration import CollaborationHub

project_manager = ProjectManager()
hub = CollaborationHub()

def register_handlers(socketio):
    @socketio.on('connect')
    def handle_connect():
        user = get_current_user()
        if user:
            hub.user_connected(user.id)
            emit('user_online', {'user_id': user.id}, broadcast=True)
            
            # Send initial state
            channels = hub.get_user_channels(user.id)
            tasks = project_manager.get_tasks(user_id=user.id)
            metrics = project_manager.get_metrics()
            
            emit('initial_state', {
                'channels': channels,
                'tasks': tasks,
                'metrics': metrics
            })

    @socketio.on('disconnect')
    def handle_disconnect():
        user = get_current_user()
        if user:
            hub.user_disconnected(user.id)
            emit('user_offline', {'user_id': user.id}, broadcast=True)

    @socketio.on('typing')
    def handle_typing(data):
        user = get_current_user()
        if user and 'channel_id' in data:
            emit('user_typing', {
                'user_id': user.id,
                'channel_id': data['channel_id']
            }, broadcast=True)

    @socketio.on('task_update')
    def handle_task_update(data):
        user = get_current_user()
        if user:
            task = project_manager.update_task(data['task_id'], data['updates'])
            emit('task_updated', task, broadcast=True)

    @socketio.on('message')
    def handle_message(data):
        user = get_current_user()
        if user and 'channel_id' in data and 'content' in data:
            message = hub.send_message(
                channel_id=data['channel_id'],
                sender=user.id,
                content=data['content'],
                thread_id=data.get('thread_id'),
                mentions=data.get('mentions', []),
                attachments=data.get('attachments', [])
            )
            emit('new_message', message, broadcast=True)

    @socketio.on('reaction')
    def handle_reaction(data):
        user = get_current_user()
        if user and 'message_id' in data and 'reaction' in data:
            hub.add_reaction(
                message_id=data['message_id'],
                user_id=user.id,
                reaction=data['reaction']
            )
            emit('message_reaction', {
                'message_id': data['message_id'],
                'user_id': user.id,
                'reaction': data['reaction']
            }, broadcast=True)

    @socketio.on('document_edit')
    def handle_document_edit(data):
        user = get_current_user()
        if user and 'doc_id' in data and 'content' in data:
            hub.update_document(
                doc_id=data['doc_id'],
                content=data['content'],
                user_id=user.id
            )
            emit('document_updated', {
                'doc_id': data['doc_id'],
                'content': data['content'],
                'editor': user.id
            }, broadcast=True)

    @socketio.on('join_channel')
    def handle_join_channel(data):
        user = get_current_user()
        if user and 'channel_id' in data:
            channel = hub.join_channel(data['channel_id'], user.id)
            emit('channel_joined', {
                'channel': channel,
                'user_id': user.id
            }, broadcast=True)

    @socketio.on('leave_channel')
    def handle_leave_channel(data):
        user = get_current_user()
        if user and 'channel_id' in data:
            hub.leave_channel(data['channel_id'], user.id)
            emit('channel_left', {
                'channel_id': data['channel_id'],
                'user_id': user.id
            }, broadcast=True)

    @socketio.on('start_thread')
    def handle_start_thread(data):
        user = get_current_user()
        if user and 'channel_id' in data and 'content' in data:
            thread = hub.create_thread(
                channel_id=data['channel_id'],
                creator=user.id,
                content=data['content']
            )
            emit('thread_started', thread, broadcast=True)

    @socketio.on('pin_message')
    def handle_pin_message(data):
        user = get_current_user()
        if user and 'message_id' in data:
            hub.pin_message(data['message_id'])
            emit('message_pinned', {
                'message_id': data['message_id'],
                'user_id': user.id
            }, broadcast=True)

    @socketio.on('request_metrics')
    def handle_metrics_request():
        user = get_current_user()
        if user:
            project_metrics = project_manager.get_metrics()
            collab_metrics = hub.get_channel_metrics()
            
            emit('metrics_update', {
                'project': project_metrics,
                'collaboration': collab_metrics
            })
