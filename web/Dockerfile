# Use an official Python runtime as the base image
FROM python:3.8

LABEL maintainer="Cuong. Duong Manh <cuongdm8499@gmail.com>" name="sentiment-analysis" build-date="20231010"

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Expose the port the application runs on
EXPOSE 8000

# Run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
