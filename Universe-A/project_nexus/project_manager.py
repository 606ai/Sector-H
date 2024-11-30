from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from enum import Enum
import uuid
from langchain import LLMChain, PromptTemplate
from langchain_community.chat_models import ChatOllama
import numpy as np

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    BLOCKED = "blocked"

@dataclass
class TeamMember:
    id: str
    name: str
    role: str
    skills: List[str]
    availability: float  # Percentage of time available
    current_tasks: List[str]
    performance_metrics: Dict[str, float]

@dataclass
class Task:
    id: str
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    assigned_to: Optional[str]
    created_at: datetime
    deadline: datetime
    estimated_hours: float
    actual_hours: float
    dependencies: List[str]
    skills_required: List[str]
    code_changes: List[str]
    reviews: List[Dict]
    comments: List[Dict]

class ProjectManager:
    def __init__(self):
        # In-memory storage for development
        self.tasks = {}
        self.documents = {}
        self.metrics = {
            'taskCompletion': 0.75,
            'teamUtilization': 0.85,
            'projectVelocity': 42
        }
        self.team_members = {}
        self.ai_agent = ChatOllama(model="llama2")
        self.task_assignment_chain = self._create_task_assignment_chain()
        self.code_review_chain = self._create_code_review_chain()
        
    def _create_task_assignment_chain(self) -> LLMChain:
        template = """
        Given the following task and team members, suggest the best team member to assign:
        
        Task: {task_description}
        Required Skills: {required_skills}
        Priority: {priority}
        Estimated Hours: {estimated_hours}
        
        Available Team Members:
        {team_members}
        
        Consider:
        1. Skill match
        2. Current workload
        3. Past performance
        4. Task priority
        
        Respond with the ID of the most suitable team member and a brief explanation.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["task_description", "required_skills", "priority", 
                           "estimated_hours", "team_members"]
        )
        
        return LLMChain(llm=self.ai_agent, prompt=prompt)
    
    def _create_code_review_chain(self) -> LLMChain:
        template = """
        Review the following code changes for potential issues:
        
        Code Changes:
        {code_changes}
        
        Consider:
        1. Code quality and best practices
        2. Potential bugs
        3. Performance implications
        4. Security concerns
        
        Provide a detailed review with specific recommendations.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["code_changes"]
        )
        
        return LLMChain(llm=self.ai_agent, prompt=prompt)

    def get_tasks(self, user):
        # In production, filter tasks by user access
        return list(self.tasks.values())

    def create_task(self, user, task_data):
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'title': task_data['title'],
            'description': task_data.get('description', ''),
            'status': task_data.get('status', 'todo'),
            'priority': task_data.get('priority', 'medium'),
            'assignee': task_data.get('assignee'),
            'created_by': user.id,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        self.tasks[task_id] = task
        return task

    def update_task(self, user, task_id, task_data):
        if task_id not in self.tasks:
            raise ValueError('Task not found')
        
        task = self.tasks[task_id]
        task.update({
            'title': task_data.get('title', task['title']),
            'description': task_data.get('description', task['description']),
            'status': task_data.get('status', task['status']),
            'priority': task_data.get('priority', task['priority']),
            'assignee': task_data.get('assignee', task['assignee']),
            'updated_at': datetime.utcnow().isoformat()
        })
        return task

    def get_documents(self, user):
        # In production, filter documents by user access
        return list(self.documents.values())

    def create_document(self, user, doc_data):
        doc_id = str(uuid.uuid4())
        document = {
            'id': doc_id,
            'title': doc_data['title'],
            'content': doc_data['content'],
            'tags': doc_data.get('tags', []),
            'created_by': user.id,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'shared_with': doc_data.get('shared_with', [])
        }
        self.documents[doc_id] = document
        return document

    def update_document(self, user, doc_id, doc_data):
        if doc_id not in self.documents:
            raise ValueError('Document not found')
        
        document = self.documents[doc_id]
        document.update({
            'title': doc_data.get('title', document['title']),
            'content': doc_data.get('content', document['content']),
            'tags': doc_data.get('tags', document['tags']),
            'shared_with': doc_data.get('shared_with', document['shared_with']),
            'updated_at': datetime.utcnow().isoformat()
        })
        return document

    def get_metrics(self, user):
        # In production, calculate real metrics based on user's projects
        return self.metrics

    def search_tasks(self, user, query):
        # Simple case-insensitive search
        query = query.lower()
        return [
            task for task in self.tasks.values()
            if query in task['title'].lower() or query in task['description'].lower()
        ]

    def add_team_member(self, name: str, role: str, skills: List[str]) -> str:
        member_id = str(uuid.uuid4())
        self.team_members[member_id] = TeamMember(
            id=member_id,
            name=name,
            role=role,
            skills=skills,
            availability=100.0,
            current_tasks=[],
            performance_metrics={
                "task_completion_rate": 1.0,
                "code_quality": 0.9,
                "collaboration_score": 0.9
            }
        )
        return member_id

    def create_task_with_details(self, title: str, description: str, priority: TaskPriority,
                   deadline: datetime, estimated_hours: float,
                   skills_required: List[str], dependencies: List[str] = None) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority,
            status=TaskStatus.TODO,
            assigned_to=None,
            created_at=datetime.now(),
            deadline=deadline,
            estimated_hours=estimated_hours,
            actual_hours=0.0,
            dependencies=dependencies or [],
            skills_required=skills_required,
            code_changes=[],
            reviews=[],
            comments=[]
        )
        
        # Auto-assign task if possible
        self._auto_assign_task(task_id)
        return task_id

    async def _auto_assign_task(self, task_id: str):
        task = self.tasks[task_id]
        
        # Prepare team member data for AI
        team_data = []
        for member in self.team_members.values():
            if member.availability > 20:  # Only consider members with >20% availability
                team_data.append({
                    "id": member.id,
                    "skills": member.skills,
                    "availability": member.availability,
                    "performance": member.performance_metrics
                })

        # Get AI recommendation
        response = await self.task_assignment_chain.arun(
            task_description=task.description,
            required_skills=task.skills_required,
            priority=task.priority.value,
            estimated_hours=task.estimated_hours,
            team_members=json.dumps(team_data, indent=2)
        )
        
        # Parse response and assign task
        assigned_member_id = response.split("\n")[0].strip()
        if assigned_member_id in self.team_members:
            self.assign_task(task_id, assigned_member_id)

    def assign_task(self, task_id: str, member_id: str):
        if task_id not in self.tasks or member_id not in self.team_members:
            raise ValueError("Invalid task_id or member_id")
            
        task = self.tasks[task_id]
        member = self.team_members[member_id]
        
        # Update task
        task.assigned_to = member_id
        task.status = TaskStatus.IN_PROGRESS
        
        # Update team member
        member.current_tasks.append(task_id)
        member.availability -= (task.estimated_hours / 40) * 100  # Assuming 40-hour work week

    async def submit_code_review(self, task_id: str, code_changes: List[str]):
        if task_id not in self.tasks:
            raise ValueError("Invalid task_id")
            
        task = self.tasks[task_id]
        task.code_changes.extend(code_changes)
        
        # Get AI code review
        review = await self.code_review_chain.arun(
            code_changes="\n".join(code_changes)
        )
        
        # Add review to task
        task.reviews.append({
            "reviewer": "AI Assistant",
            "timestamp": datetime.now().isoformat(),
            "content": review
        })
        
        task.status = TaskStatus.REVIEW
        return review

    def update_task_status(self, task_id: str, status: TaskStatus,
                          actual_hours: float = None):
        if task_id not in self.tasks:
            raise ValueError("Invalid task_id")
            
        task = self.tasks[task_id]
        task.status = status
        
        if actual_hours is not None:
            task.actual_hours = actual_hours
            
        if status == TaskStatus.COMPLETED:
            # Update team member metrics
            if task.assigned_to:
                member = self.team_members[task.assigned_to]
                member.current_tasks.remove(task_id)
                member.availability += (task.estimated_hours / 40) * 100
                
                # Update performance metrics
                efficiency = task.estimated_hours / task.actual_hours
                member.performance_metrics["task_completion_rate"] = (
                    0.7 * member.performance_metrics["task_completion_rate"] +
                    0.3 * efficiency
                )

    def get_project_metrics(self) -> Dict:
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for t in self.tasks.values() 
                            if t.status == TaskStatus.COMPLETED)
        overdue_tasks = sum(1 for t in self.tasks.values()
                           if t.deadline < datetime.now() and 
                           t.status != TaskStatus.COMPLETED)
        
        # Calculate burndown and velocity
        velocity = []
        for week in range(4):  # Last 4 weeks
            start_date = datetime.now() - timedelta(weeks=week+1)
            end_date = datetime.now() - timedelta(weeks=week)
            completed_in_week = sum(
                1 for t in self.tasks.values()
                if t.status == TaskStatus.COMPLETED and
                start_date <= t.created_at <= end_date
            )
            velocity.append(completed_in_week)
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "overdue_tasks": overdue_tasks,
            "average_velocity": np.mean(velocity),
            "team_utilization": np.mean([m.availability for m in self.team_members.values()]),
            "skill_coverage": self._calculate_skill_coverage()
        }

    def _calculate_skill_coverage(self) -> Dict[str, int]:
        skill_coverage = {}
        for member in self.team_members.values():
            for skill in member.skills:
                skill_coverage[skill] = skill_coverage.get(skill, 0) + 1
        return skill_coverage

    def get_task_dependencies_graph(self) -> Dict:
        """Generate a dependency graph for visualization"""
        nodes = []
        edges = []
        
        for task in self.tasks.values():
            nodes.append({
                "id": task.id,
                "label": task.title,
                "status": task.status.value,
                "priority": task.priority.value
            })
            
            for dep in task.dependencies:
                edges.append({
                    "from": dep,
                    "to": task.id
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
