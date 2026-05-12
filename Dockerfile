FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies (optional, but useful for numpy/scipy etc.)
RUN apt-get update && apt-get install -y \
    build-essential python3-dev git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip install jupyter

# Copy project files
COPY . .

CMD ["bash"]
