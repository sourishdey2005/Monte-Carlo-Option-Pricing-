# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && useradd -m -u 1000 user \
    && rm -rf /var/lib/apt/lists/*
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the requirements file into the container
COPY --chown=user requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the current directory contents into the container
COPY --chown=user . .

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py"]
