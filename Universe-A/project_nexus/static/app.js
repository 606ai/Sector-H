const app = Vue.createApp({
    data() {
        return {
            tasks: [],
            team: [],
            channels: [],
            documents: [],
            currentUser: null,
            activeView: 'tasks',
            selectedTask: null,
            selectedChannel: null,
            newMessage: '',
            newDocument: {
                title: '',
                content: '',
                tags: []
            },
            metrics: {
                taskCompletion: 0,
                teamUtilization: 0,
                projectVelocity: 0
            },
            notifications: [],
            socket: null
        }
    },
    
    computed: {
        sortedTasks() {
            return [...this.tasks].sort((a, b) => {
                const priorityOrder = {
                    'critical': 0,
                    'high': 1,
                    'medium': 2,
                    'low': 3
                };
                return priorityOrder[a.priority] - priorityOrder[b.priority];
            });
        },
        
        activeThreads() {
            if (!this.selectedChannel) return [];
            return this.channels.find(c => c.id === this.selectedChannel)
                ?.messages.filter(m => m.thread_id) || [];
        },
        
        tasksByStatus() {
            const statusGroups = {
                'todo': [],
                'in_progress': [],
                'review': [],
                'completed': []
            };
            
            this.tasks.forEach(task => {
                if (statusGroups[task.status]) {
                    statusGroups[task.status].push(task);
                }
            });
            
            return statusGroups;
        }
    },
    
    methods: {
        async initialize() {
            this.socket = io();
            await this.setupWebSocket();
            await this.loadInitialData();
            this.startMetricsPolling();
        },
        
        async setupWebSocket() {
            this.socket.on('task_update', this.handleTaskUpdate);
            this.socket.on('new_message', this.handleNewMessage);
            this.socket.on('metrics_update', this.handleMetricsUpdate);
        },
        
        async loadInitialData() {
            try {
                const [tasks, team, channels, documents] = await Promise.all([
                    this.fetchTasks(),
                    this.fetchTeam(),
                    this.fetchChannels(),
                    this.fetchDocuments()
                ]);
                
                this.tasks = tasks;
                this.team = team;
                this.channels = channels;
                this.documents = documents;
            } catch (error) {
                console.error('Error loading initial data:', error);
                this.showNotification('Error loading data', 'error');
            }
        },
        
        async fetchTasks() {
            const response = await fetch('/api/tasks');
            if (!response.ok) throw new Error('Failed to fetch tasks');
            return response.json();
        },
        
        async createTask(taskData) {
            try {
                const response = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(taskData)
                });
                
                if (!response.ok) throw new Error('Failed to create task');
                
                const task = await response.json();
                this.tasks.push(task);
                this.showNotification('Task created successfully', 'success');
            } catch (error) {
                console.error('Error creating task:', error);
                this.showNotification('Error creating task', 'error');
            }
        },
        
        async updateTaskStatus(taskId, newStatus) {
            try {
                const response = await fetch(`/api/tasks/${taskId}/status`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ status: newStatus })
                });
                
                if (!response.ok) throw new Error('Failed to update task status');
                
                const updatedTask = await response.json();
                const index = this.tasks.findIndex(t => t.id === taskId);
                if (index !== -1) {
                    this.tasks[index] = updatedTask;
                }
                
                this.showNotification('Task status updated', 'success');
            } catch (error) {
                console.error('Error updating task status:', error);
                this.showNotification('Error updating task status', 'error');
            }
        },
        
        async sendMessage(channelId, content, threadId = null) {
            try {
                const response = await fetch(`/api/channels/${channelId}/messages`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        content,
                        thread_id: threadId
                    })
                });
                
                if (!response.ok) throw new Error('Failed to send message');
                
                const message = await response.json();
                const channel = this.channels.find(c => c.id === channelId);
                if (channel) {
                    channel.messages.push(message);
                }
                
                this.newMessage = '';
            } catch (error) {
                console.error('Error sending message:', error);
                this.showNotification('Error sending message', 'error');
            }
        },
        
        async createDocument() {
            try {
                const response = await fetch('/api/documents', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.newDocument)
                });
                
                if (!response.ok) throw new Error('Failed to create document');
                
                const document = await response.json();
                this.documents.push(document);
                this.newDocument = { title: '', content: '', tags: [] };
                this.showNotification('Document created successfully', 'success');
            } catch (error) {
                console.error('Error creating document:', error);
                this.showNotification('Error creating document', 'error');
            }
        },
        
        startMetricsPolling() {
            setInterval(async () => {
                try {
                    const response = await fetch('/api/metrics');
                    if (!response.ok) throw new Error('Failed to fetch metrics');
                    
                    const metrics = await response.json();
                    this.metrics = metrics;
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                }
            }, 30000); // Poll every 30 seconds
        },
        
        handleTaskUpdate(task) {
            const index = this.tasks.findIndex(t => t.id === task.id);
            if (index !== -1) {
                this.tasks[index] = task;
            } else {
                this.tasks.push(task);
            }
        },
        
        handleNewMessage(message) {
            const channel = this.channels.find(c => c.id === message.channel_id);
            if (channel) {
                channel.messages.push(message);
                if (this.selectedChannel === message.channel_id) {
                    this.$nextTick(() => {
                        this.scrollToBottom();
                    });
                }
            }
        },
        
        handleMetricsUpdate(metrics) {
            this.metrics = metrics;
        },
        
        showNotification(message, type = 'info') {
            const notification = {
                id: Date.now(),
                message,
                type
            };
            
            this.notifications.push(notification);
            setTimeout(() => {
                const index = this.notifications.findIndex(n => n.id === notification.id);
                if (index !== -1) {
                    this.notifications.splice(index, 1);
                }
            }, 5000);
        },
        
        scrollToBottom() {
            const messageContainer = this.$refs.messageContainer;
            if (messageContainer) {
                messageContainer.scrollTop = messageContainer.scrollHeight;
            }
        }
    },
    
    created() {
        this.initialize();
    }
});

app.mount('#app');
