# Base image — Python 3.12 matches what you're already using
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (faster rebuilds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else into the container
COPY . .

# Command to run when the container starts
CMD ["python", "main.py"]