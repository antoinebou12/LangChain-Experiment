# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt and wget
RUN apt-get update && \
    apt-get install -y wget && \
    pip install -r requirements.txt && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-gpt4all-j-v1.2-jazzy.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-gpt4all-j-v1.1-breezy.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-gpt4all-j.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-vicuna-7b-1.1-q4_2.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-vicuna-13b-1.1-q4_2.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-wizardLM-7B.q4_2.bin && \
    wget -P /app/models/ https://gpt4all.io/models/ggml-stable-vicuna-13B.q4_2.bin

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port", "80", "--server.headless", "true"]
