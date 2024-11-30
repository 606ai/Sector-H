from typing import Dict, List, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass
import json
import uuid
from langchain import LLMChain, PromptTemplate
from langchain_community.chat_models import ChatOllama

@dataclass
class Message:
    id: str
    sender: str
    content: str
    timestamp: datetime
    thread_id: Optional[str]
    mentions: List[str]
    attachments: List[str]
    reactions: Dict[str, List[str]]

@dataclass
class Document:
    id: str
    title: str
    content: str
    created_by: str
    created_at: datetime
    last_modified: datetime
    version: int
    shared_with: List[str]
    comments: List[Dict]
    tags: List[str]

class CollaborationHub:
    def __init__(self):
        self.channels = {}  # Channel ID -> List of Messages
        self.documents = {}  # Document ID -> Document
        self.active_users = set()
        self.ai_agent = ChatOllama(model="llama2")
        self.message_analysis_chain = self._create_message_analysis_chain()
        self.channels_storage = {}
        self.messages_storage = {}
        self.threads_storage = {}
        self.document_comments_storage = {}

    def _create_message_analysis_chain(self) -> LLMChain:
        template = """
        Analyze the following conversation for action items and key decisions:
        
        Conversation:
        {conversation}
        
        Extract:
        1. Action items (tasks that need to be done)
        2. Decisions made
        3. Important updates or announcements
        4. Questions that need follow-up
        
        Format the response as JSON with these categories.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["conversation"]
        )
        
        return LLMChain(llm=self.ai_agent, prompt=prompt)

    def create_channel(self, name: str, description: str, members: List[str]) -> str:
        channel_id = str(uuid.uuid4())
        self.channels[channel_id] = {
            "name": name,
            "description": description,
            "members": members,
            "messages": [],
            "pinned_messages": [],
            "created_at": datetime.now()
        }
        return channel_id

    async def send_message(self, channel_id: str, sender: str, content: str,
                          thread_id: str = None, mentions: List[str] = None,
                          attachments: List[str] = None) -> Message:
        if channel_id not in self.channels:
            raise ValueError("Invalid channel_id")
            
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender,
            content=content,
            timestamp=datetime.now(),
            thread_id=thread_id,
            mentions=mentions or [],
            attachments=attachments or [],
            reactions={}
        )
        
        self.channels[channel_id]["messages"].append(message)
        
        # Analyze conversation if it's a thread
        if thread_id:
            thread_messages = [
                msg for msg in self.channels[channel_id]["messages"]
                if msg.thread_id == thread_id
            ]
            conversation = "\n".join([
                f"{msg.sender}: {msg.content}" for msg in thread_messages
            ])
            
            analysis = await self.message_analysis_chain.arun(
                conversation=conversation
            )
            
            # Convert analysis to structured data
            try:
                analysis_data = json.loads(analysis)
                # Store analysis with the thread
                self.channels[channel_id]["thread_analysis"] = {
                    "thread_id": thread_id,
                    "last_updated": datetime.now(),
                    "data": analysis_data
                }
            except json.JSONDecodeError:
                pass  # Handle invalid JSON response
        
        return message

    def add_reaction(self, channel_id: str, message_id: str,
                    user_id: str, reaction: str):
        channel = self.channels.get(channel_id)
        if not channel:
            raise ValueError("Invalid channel_id")
            
        message = next(
            (msg for msg in channel["messages"] if msg.id == message_id),
            None
        )
        if not message:
            raise ValueError("Invalid message_id")
            
        if reaction not in message.reactions:
            message.reactions[reaction] = []
        
        if user_id not in message.reactions[reaction]:
            message.reactions[reaction].append(user_id)

    def create_document(self, title: str, content: str, created_by: str,
                       shared_with: List[str], tags: List[str] = None) -> str:
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = Document(
            id=doc_id,
            title=title,
            content=content,
            created_by=created_by,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            version=1,
            shared_with=shared_with,
            comments=[],
            tags=tags or []
        )
        return doc_id

    def update_document(self, doc_id: str, content: str, user_id: str):
        if doc_id not in self.documents:
            raise ValueError("Invalid document_id")
            
        doc = self.documents[doc_id]
        if user_id not in [doc.created_by] + doc.shared_with:
            raise ValueError("User does not have permission to edit this document")
            
        doc.content = content
        doc.last_modified = datetime.now()
        doc.version += 1

    def add_document_comment(self, doc_id: str, user_id: str,
                           content: str, position: str = None):
        if doc_id not in self.documents:
            raise ValueError("Invalid document_id")
            
        doc = self.documents[doc_id]
        if user_id not in [doc.created_by] + doc.shared_with:
            raise ValueError("User does not have permission to comment on this document")
            
        comment = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "content": content,
            "position": position,
            "timestamp": datetime.now(),
            "replies": []
        }
        
        doc.comments.append(comment)
        return comment["id"]

    def get_channel_summary(self, channel_id: str, days: int = 7) -> Dict:
        if channel_id not in self.channels:
            raise ValueError("Invalid channel_id")
            
        channel = self.channels[channel_id]
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_messages = [
            msg for msg in channel["messages"]
            if msg.timestamp >= cutoff_date
        ]
        
        return {
            "total_messages": len(recent_messages),
            "active_threads": len(set(
                msg.thread_id for msg in recent_messages
                if msg.thread_id is not None
            )),
            "active_users": len(set(
                msg.sender for msg in recent_messages
            )),
            "popular_threads": self._get_popular_threads(recent_messages),
            "engagement_metrics": self._calculate_engagement_metrics(recent_messages)
        }

    def _get_popular_threads(self, messages: List[Message]) -> List[Dict]:
        thread_stats = {}
        
        for msg in messages:
            if msg.thread_id:
                if msg.thread_id not in thread_stats:
                    thread_stats[msg.thread_id] = {
                        "message_count": 0,
                        "participant_count": set(),
                        "reaction_count": 0
                    }
                    
                stats = thread_stats[msg.thread_id]
                stats["message_count"] += 1
                stats["participant_count"].add(msg.sender)
                stats["reaction_count"] += sum(
                    len(users) for users in msg.reactions.values()
                )
        
        # Convert to list and sort by engagement
        popular_threads = [
            {
                "thread_id": thread_id,
                "message_count": stats["message_count"],
                "participant_count": len(stats["participant_count"]),
                "reaction_count": stats["reaction_count"]
            }
            for thread_id, stats in thread_stats.items()
        ]
        
        return sorted(
            popular_threads,
            key=lambda x: (x["message_count"], x["participant_count"], x["reaction_count"]),
            reverse=True
        )[:5]  # Return top 5

    def _calculate_engagement_metrics(self, messages: List[Message]) -> Dict:
        total_messages = len(messages)
        if total_messages == 0:
            return {
                "avg_reactions_per_message": 0,
                "avg_thread_length": 0,
                "mention_frequency": 0
            }
            
        total_reactions = sum(
            sum(len(users) for users in msg.reactions.values())
            for msg in messages
        )
        
        thread_messages = [msg for msg in messages if msg.thread_id]
        thread_count = len(set(msg.thread_id for msg in thread_messages))
        
        total_mentions = sum(len(msg.mentions) for msg in messages)
        
        return {
            "avg_reactions_per_message": total_reactions / total_messages,
            "avg_thread_length": len(thread_messages) / thread_count if thread_count > 0 else 0,
            "mention_frequency": total_mentions / total_messages
        }

    def get_channels(self, user):
        # In production, filter channels by user access
        return list(self.channels_storage.values())

    def create_channel(self, user, channel_data):
        channel_id = str(uuid.uuid4())
        channel = {
            'id': channel_id,
            'name': channel_data['name'],
            'description': channel_data.get('description', ''),
            'created_by': user.id,
            'created_at': datetime.utcnow().isoformat(),
            'members': channel_data.get('members', [user.id]),
            'is_private': channel_data.get('is_private', False)
        }
        self.channels_storage[channel_id] = channel
        return channel

    def get_messages(self, user, channel_id):
        if channel_id not in self.channels_storage:
            raise ValueError('Channel not found')
        
        # In production, verify user has access to channel
        return [
            msg for msg in self.messages_storage.values()
            if msg['channel_id'] == channel_id
        ]

    def send_message(self, user, channel_id, message_data):
        if channel_id not in self.channels_storage:
            raise ValueError('Channel not found')
        
        message_id = str(uuid.uuid4())
        message = {
            'id': message_id,
            'channel_id': channel_id,
            'content': message_data['content'],
            'sender': user.id,
            'sender_name': user.username,  # For display purposes
            'timestamp': datetime.utcnow().isoformat(),
            'thread_id': message_data.get('thread_id'),
            'mentions': message_data.get('mentions', []),
            'attachments': message_data.get('attachments', [])
        }
        
        # If this is a threaded message, update thread
        if message['thread_id']:
            if message['thread_id'] not in self.threads_storage:
                self.threads_storage[message['thread_id']] = {
                    'id': message['thread_id'],
                    'channel_id': channel_id,
                    'messages': []
                }
            self.threads_storage[message['thread_id']]['messages'].append(message_id)
        
        self.messages_storage[message_id] = message
        return message

    def add_document_comment(self, user, doc_id, comment_data):
        comment_id = str(uuid.uuid4())
        comment = {
            'id': comment_id,
            'doc_id': doc_id,
            'content': comment_data['content'],
            'position': comment_data.get('position'),  # For inline comments
            'created_by': user.id,
            'created_at': datetime.utcnow().isoformat(),
            'resolved': False
        }
        
        if doc_id not in self.document_comments_storage:
            self.document_comments_storage[doc_id] = []
        
        self.document_comments_storage[doc_id].append(comment)
        return comment

    def get_channel_metrics(self):
        total_messages = len(self.messages_storage)
        total_threads = len(self.threads_storage)
        active_channels = len(self.channels_storage)
        
        return {
            'totalMessages': total_messages,
            'totalThreads': total_threads,
            'activeChannels': active_channels,
            'averageThreadLength': total_threads and total_messages / total_threads or 0
        }

    def search_messages(self, user, query):
        # Simple case-insensitive search
        query = query.lower()
        return [
            msg for msg in self.messages_storage.values()
            if query in msg['content'].lower()
        ]

    def search_documents(self, user, query):
        # This would typically search document content and metadata
        query = query.lower()
        return []  # Implement document search when document storage is added
