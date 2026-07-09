# ============================================================
# PapsAI XNet Dockerfile
# Explainable Hybrid Deep Learning for Automated
# Cervical Cytology Classification
# ============================================================

FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, image processing, and scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create common runtime directories
RUN mkdir -p data datasets outputs figures logs checkpoints

# Expose Jupyter port if notebooks are used
EXPOSE 8888

# Default command
CMD ["python", "inference.py"]
