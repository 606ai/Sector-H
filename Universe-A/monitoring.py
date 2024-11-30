from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import psutil
import logging
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import json
import threading
from datetime import datetime

@dataclass
class SimulationMetrics:
    agents_count: int = 0
    active_scenarios: int = 0
    events_processed: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0

class MetricsCollector:
    def __init__(self):
        # Prometheus metrics
        self.active_users = Gauge('universe_a_active_users', 'Number of active users')
        self.active_scenarios = Gauge('universe_a_active_scenarios', 'Number of active scenarios')
        self.agents_total = Gauge('universe_a_agents_total', 'Total number of agents')
        self.events_processed = Counter('universe_a_events_processed', 'Total events processed')
        self.response_time = Histogram('universe_a_response_time', 'Response time in seconds')
        
        # Performance metrics
        self.cpu_usage = Gauge('universe_a_cpu_usage', 'CPU usage percentage')
        self.memory_usage = Gauge('universe_a_memory_usage', 'Memory usage in MB')
        
        # Start Prometheus HTTP server
        start_http_server(8000)
        
        # Initialize logging
        logging.basicConfig(
            filename='universe_a_metrics.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('UniverseA-Metrics')
        
        # Start metrics collection thread
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.start()

    def _collect_metrics(self):
        while self.running:
            try:
                # Collect system metrics
                cpu = psutil.cpu_percent()
                memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Update Prometheus metrics
                self.cpu_usage.set(cpu)
                self.memory_usage.set(memory)
                
                # Log metrics
                self.logger.info(f"System Metrics - CPU: {cpu}%, Memory: {memory:.2f}MB")
                
                time.sleep(15)  # Collect every 15 seconds
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")

    def record_event(self, event_type: str, data: Dict):
        """Record a simulation event"""
        try:
            self.events_processed.inc()
            
            # Log event
            self.logger.info(f"Event: {event_type} - {json.dumps(data)}")
        except Exception as e:
            self.logger.error(f"Error recording event: {str(e)}")

    def update_scenario_metrics(self, scenario_id: str, metrics: SimulationMetrics):
        """Update metrics for a specific scenario"""
        try:
            self.active_scenarios.set(metrics.active_scenarios)
            self.agents_total.set(metrics.agents_count)
            
            # Log scenario metrics
            self.logger.info(f"Scenario {scenario_id} Metrics: {metrics}")
        except Exception as e:
            self.logger.error(f"Error updating scenario metrics: {str(e)}")

    def track_response_time(self, func):
        """Decorator to track function response time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            self.response_time.observe(duration)
            return result
        return wrapper

class AnalyticsManager:
    def __init__(self):
        self.session_data = []
        self.agent_behaviors = {}
        self.scenario_stats = {}
        
    def record_session(self, user_id: str, scenario_type: str, duration: int):
        """Record user session data"""
        session = {
            'user_id': user_id,
            'scenario_type': scenario_type,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        self.session_data.append(session)
        
    def record_agent_behavior(self, agent_id: str, behavior_data: Dict):
        """Record agent behavior patterns"""
        if agent_id not in self.agent_behaviors:
            self.agent_behaviors[agent_id] = []
        self.agent_behaviors[agent_id].append({
            'timestamp': datetime.now().isoformat(),
            **behavior_data
        })
        
    def record_scenario_stats(self, scenario_id: str, stats: Dict):
        """Record scenario statistics"""
        if scenario_id not in self.scenario_stats:
            self.scenario_stats[scenario_id] = []
        self.scenario_stats[scenario_id].append({
            'timestamp': datetime.now().isoformat(),
            **stats
        })
        
    def generate_report(self, start_time: datetime, end_time: datetime) -> Dict:
        """Generate analytics report for a time period"""
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'sessions': {
                'total_count': 0,
                'average_duration': 0,
                'scenario_distribution': {}
            },
            'agent_behaviors': {
                'most_common_actions': {},
                'average_success_rate': 0
            },
            'scenario_performance': {
                'completion_rates': {},
                'average_duration': {}
            }
        }
        
        # Process session data
        relevant_sessions = [
            s for s in self.session_data 
            if start_time <= datetime.fromisoformat(s['timestamp']) <= end_time
        ]
        
        if relevant_sessions:
            report['sessions']['total_count'] = len(relevant_sessions)
            report['sessions']['average_duration'] = sum(s['duration'] for s in relevant_sessions) / len(relevant_sessions)
            
            # Calculate scenario distribution
            for session in relevant_sessions:
                scenario = session['scenario_type']
                report['sessions']['scenario_distribution'][scenario] = report['sessions']['scenario_distribution'].get(scenario, 0) + 1
        
        # Process agent behaviors
        for agent_id, behaviors in self.agent_behaviors.items():
            relevant_behaviors = [
                b for b in behaviors 
                if start_time <= datetime.fromisoformat(b['timestamp']) <= end_time
            ]
            
            if relevant_behaviors:
                # Count action types
                action_counts = {}
                success_count = 0
                
                for behavior in relevant_behaviors:
                    action = behavior.get('action_type')
                    if action:
                        action_counts[action] = action_counts.get(action, 0) + 1
                    if behavior.get('success'):
                        success_count += 1
                
                report['agent_behaviors']['most_common_actions'][agent_id] = action_counts
                report['agent_behaviors']['average_success_rate'] = success_count / len(relevant_behaviors)
        
        # Process scenario stats
        for scenario_id, stats_list in self.scenario_stats.items():
            relevant_stats = [
                s for s in stats_list 
                if start_time <= datetime.fromisoformat(s['timestamp']) <= end_time
            ]
            
            if relevant_stats:
                completion_count = sum(1 for s in relevant_stats if s.get('completed'))
                total_count = len(relevant_stats)
                
                report['scenario_performance']['completion_rates'][scenario_id] = completion_count / total_count
                report['scenario_performance']['average_duration'][scenario_id] = sum(s.get('duration', 0) for s in relevant_stats) / total_count
        
        return report

    def export_data(self, format: str = 'json') -> str:
        """Export analytics data in specified format"""
        data = {
            'sessions': self.session_data,
            'agent_behaviors': self.agent_behaviors,
            'scenario_stats': self.scenario_stats
        }
        
        if format == 'json':
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def cleanup_old_data(self, days: int = 30):
        """Clean up data older than specified days"""
        cutoff_date = datetime.now() - datetime.timedelta(days=days)
        
        self.session_data = [
            s for s in self.session_data 
            if datetime.fromisoformat(s['timestamp']) > cutoff_date
        ]
        
        for agent_id in self.agent_behaviors:
            self.agent_behaviors[agent_id] = [
                b for b in self.agent_behaviors[agent_id]
                if datetime.fromisoformat(b['timestamp']) > cutoff_date
            ]
            
        for scenario_id in self.scenario_stats:
            self.scenario_stats[scenario_id] = [
                s for s in self.scenario_stats[scenario_id]
                if datetime.fromisoformat(s['timestamp']) > cutoff_date
            ]
