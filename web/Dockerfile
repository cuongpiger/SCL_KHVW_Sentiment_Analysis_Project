# Use an official Python runtime as the base image
FROM ubuntu:20.04

LABEL maintainer="Cuong. Duong Manh <cuongdm8499@gmail.com>" name="sentiment-analysis" build-date="20231011"

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install python3 python3-pip enchant-2 -y && \ 
    pip3 install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Expose the port the application runs on
EXPOSE 8000

# Run the application
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
