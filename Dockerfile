# Base Image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (optional, for future web-based interaction)
EXPOSE 5000

# Command to run the simulation script
CMD ["python", "simulate.py"]
