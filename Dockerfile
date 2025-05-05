# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some python packages
# (Example: build-essential for packages needing compilation)
# Add git for cloning repositories
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable default (can be overridden)
# Note: OPENAI_API_KEY should ideally be passed securely, not hardcoded or defaulted here.
# It will be loaded from .env file mounted via docker-compose.
ENV CHROMA_DB_DIR=/app/chroma_db_volume
ENV API_URL=http://0.0.0.0:8000

# Define the command to run the application using uvicorn
# Use 0.0.0.0 to allow connections from outside the container
# Use --reload for development if needed, but remove for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 