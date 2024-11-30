FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment files
COPY environment.yml .
COPY api/requirements.txt .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Activate conda environment
SHELL ["conda", "run", "-n", "sector-h", "/bin/bash", "-c"]

# Install additional requirements
RUN pip install -r requirements.txt

# Copy application code
COPY api/ .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["conda", "run", "--no-capture-output", "-n", "sector-h", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
