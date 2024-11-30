from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import threading
import time
from monitoring import MetricsCollector, AnalyticsManager
from datetime import datetime, timedelta

app = Flask(__name__)
socketio = SocketIO(app)

metrics_collector = MetricsCollector()
analytics_manager = AnalyticsManager()

@app.route('/dashboard')
def dashboard():
    return render_template('monitoring_dashboard.html')

@app.route('/api/metrics/current')
def current_metrics():
    return jsonify({
        'cpu_usage': metrics_collector.cpu_usage._value.get(),
        'memory_usage': metrics_collector.memory_usage._value.get(),
        'active_users': metrics_collector.active_users._value.get(),
        'active_scenarios': metrics_collector.active_scenarios._value.get()
    })

@app.route('/api/analytics/summary')
def analytics_summary():
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)  # Last 7 days
    return jsonify(analytics_manager.generate_report(start_time, end_time))

def emit_metrics():
    """Emit metrics to connected clients"""
    while True:
        metrics = {
            'cpu_usage': metrics_collector.cpu_usage._value.get(),
            'memory_usage': metrics_collector.memory_usage._value.get(),
            'active_users': metrics_collector.active_users._value.get(),
            'active_scenarios': metrics_collector.active_scenarios._value.get(),
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('metrics_update', metrics)
        time.sleep(5)

@socketio.on('connect')
def handle_connect():
    # Start metrics emission in a separate thread
    thread = threading.Thread(target=emit_metrics)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8080)
