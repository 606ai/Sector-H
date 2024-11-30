from PyQt6.QtWidgets import QWidget, QGraphicsView, QGraphicsScene
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont
import random

class WorldView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Grid settings
        self.grid_size = 40
        self.cell_size = 20
        self.agents = {}
        self.resources = {}
        self.structures = {}
        
        # Visual settings
        self.colors = {
            'background': QColor('#1a1a1a'),
            'grid': QColor('#2a2a2a'),
            'agent': QColor('#00ff00'),
            'resource': QColor('#ffff00'),
            'structure': QColor('#808080'),
            'text': QColor('#ffffff')
        }
        
        self.init_grid()

    def init_grid(self):
        """Initialize the grid"""
        self.scene.clear()
        
        # Set background
        self.setBackgroundBrush(QBrush(self.colors['background']))
        
        # Draw grid
        pen = QPen(self.colors['grid'])
        pen.setWidth(1)
        
        for i in range(self.grid_size + 1):
            # Vertical lines
            self.scene.addLine(i * self.cell_size, 0,
                             i * self.cell_size, self.grid_size * self.cell_size,
                             pen)
            # Horizontal lines
            self.scene.addLine(0, i * self.cell_size,
                             self.grid_size * self.cell_size, i * self.cell_size,
                             pen)

    def add_agent(self, agent_id, x, y, agent_type='default'):
        """Add an agent to the visualization"""
        if agent_id in self.agents:
            self.scene.removeItem(self.agents[agent_id])
            
        brush = QBrush(self.colors['agent'])
        agent_visual = self.scene.addEllipse(
            x * self.cell_size + 2,
            y * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4,
            QPen(Qt.PenStyle.NoPen),
            brush
        )
        
        # Add agent label
        text = self.scene.addText(str(agent_id)[:4], QFont("Arial", 8))
        text.setDefaultTextColor(self.colors['text'])
        text.setPos(x * self.cell_size, y * self.cell_size)
        
        self.agents[agent_id] = (agent_visual, text)

    def add_resource(self, resource_id, x, y, resource_type='default'):
        """Add a resource to the visualization"""
        if resource_id in self.resources:
            self.scene.removeItem(self.resources[resource_id])
            
        brush = QBrush(self.colors['resource'])
        resource_visual = self.scene.addRect(
            x * self.cell_size + 5,
            y * self.cell_size + 5,
            self.cell_size - 10,
            self.cell_size - 10,
            QPen(Qt.PenStyle.NoPen),
            brush
        )
        
        self.resources[resource_id] = resource_visual

    def add_structure(self, structure_id, x, y, size=1):
        """Add a structure to the visualization"""
        if structure_id in self.structures:
            self.scene.removeItem(self.structures[structure_id])
            
        brush = QBrush(self.colors['structure'])
        structure_visual = self.scene.addRect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size * size,
            self.cell_size * size,
            QPen(Qt.PenStyle.NoPen),
            brush
        )
        
        self.structures[structure_id] = structure_visual

    def update_world_state(self, world_state):
        """Update the entire world state"""
        self.scene.clear()
        self.init_grid()
        
        # Update agents
        for agent_id, position in world_state.get('agents', {}).items():
            self.add_agent(agent_id, position['x'], position['y'])
            
        # Update resources
        for resource_id, position in world_state.get('resources', {}).items():
            self.add_resource(resource_id, position['x'], position['y'])
            
        # Update structures
        for structure_id, data in world_state.get('structures', {}).items():
            self.add_structure(structure_id, data['x'], data['y'], data['size'])

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        self.fitInView(QRectF(0, 0,
                             self.grid_size * self.cell_size,
                             self.grid_size * self.cell_size),
                      Qt.AspectRatioMode.KeepAspectRatio)
