# Use an official Python runtime as a base image
FROM python:3.11.5-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
COPY mlruns /app/models
# Install the required packages
RUN pip install Flask pandas mlflow scikit-learn

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable to disable Flask's default setting to run the app in production mode
ENV FLASK_ENV=development

# Run the application
CMD ["python", "app2.py"]
