FROM jupyter/datascience-notebook:latest

WORKDIR /workspace

# Copy requirements
COPY ../requirements.txt .

# Install additional Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter port
EXPOSE 8888

# Set working directory for notebooks
WORKDIR /workspace/notebooks

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='sector-h'"]
