# Use Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /mlflow

# Install MLflow and PostgreSQL adapter
RUN pip install --no-cache-dir \
    mlflow==2.5.0 \
    psycopg2==2.9.6 \
    boto3==1.28.1

# Expose MLflow UI port
EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "${MLFLOW_TRACKING_URI}", \
     "--default-artifact-root", "/mlflow"]
