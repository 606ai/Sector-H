import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QTextEdit, QLabel, QComboBox, QHBoxLayout,
                            QDockWidget, QListWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
import requests
from pathlib import Path
import os
from dotenv import load_dotenv

from visualization import WorldView
from advanced_agent import AdvancedAgent
from scenarios import CivilizationScenario
from world import World

class UniverseSimulation(QThread):
    update_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.world = World()
        self.scenario = None
        self.tick_rate = 1000  # milliseconds

    def run(self):
        self.running = True
        while self.running:
            if self.scenario:
                update = self.scenario.update()
                self.update_signal.emit(update)
            self.msleep(self.tick_rate)

    def stop(self):
        self.running = False

    def set_scenario(self, scenario_type):
        if scenario_type == "civilization":
            self.scenario = CivilizationScenario()
            self.scenario.initialize()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universe-A")
        self.setMinimumSize(1200, 800)
        
        # Create central widget with main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create left control panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

        # Model selection
        model_label = QLabel("Select AI Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["llama2", "mistral", "codellama"])
        
        # Scenario selection
        scenario_label = QLabel("Select Scenario:")
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(["civilization", "exploration", "competition"])

        # Control buttons
        self.start_button = QPushButton("Start Simulation")
        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setEnabled(False)

        # Add widgets to left panel
        left_layout.addWidget(model_label)
        left_layout.addWidget(self.model_combo)
        left_layout.addWidget(scenario_label)
        left_layout.addWidget(self.scenario_combo)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.stop_button)
        left_layout.addStretch()

        # Create world visualization
        self.world_view = WorldView()

        # Create right info panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMaximumWidth(300)

        # Add info widgets to right panel
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.event_list = QListWidget()
        
        right_layout.addWidget(QLabel("World Information:"))
        right_layout.addWidget(self.info_text)
        right_layout.addWidget(QLabel("Recent Events:"))
        right_layout.addWidget(self.event_list)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.world_view, stretch=1)
        main_layout.addWidget(right_panel)

        # Connect buttons
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)

        # Initialize simulation thread
        self.simulation = UniverseSimulation()
        self.simulation.update_signal.connect(self.update_world_state)

    def start_simulation(self):
        scenario_type = self.scenario_combo.currentText()
        self.simulation.set_scenario(scenario_type)
        self.simulation.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.model_combo.setEnabled(False)
        self.scenario_combo.setEnabled(False)

    def stop_simulation(self):
        self.simulation.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.model_combo.setEnabled(True)
        self.scenario_combo.setEnabled(True)

    def update_world_state(self, state):
        # Update visualization
        self.world_view.update_world_state(state.get('state', {}))
        
        # Update info panel
        info_text = f"""
        Time: {state.get('time', 0)}
        Technology Level: {state.get('state', {}).get('technology_level', 0)}
        Resources:
        """
        resources = state.get('state', {}).get('resources', {})
        for resource, amount in resources.items():
            info_text += f"  {resource}: {amount:.1f}\n"
        
        self.info_text.setText(info_text)
        
        # Update event list
        self.event_list.clear()
        for event in state.get('state', {}).get('events', []):
            self.event_list.addItem(f"{event['type']}: {event['description']}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
