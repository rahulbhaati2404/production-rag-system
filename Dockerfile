# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install system dependencies (needed for certain AI libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Create directories for data and DB persistence
RUN mkdir -p data/processed vector_db

# 7. Environment variable to ensure output isn't buffered (better logging)
ENV PYTHONUNBUFFERED=1

# 8. Command to run (we default to the agent, but this can be overridden)
CMD ["python", "03_retrieval_agent.py"]