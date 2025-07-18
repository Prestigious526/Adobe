# Use official Python image
FROM --platform=linux/amd64 python:3.10-slim

# Create app directory
WORKDIR /app

# Copy files
COPY requirements.txt ./
COPY main.py ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make input/output directories
RUN mkdir /app/input /app/output

# Run when container starts
CMD ["python", "main.py"]
