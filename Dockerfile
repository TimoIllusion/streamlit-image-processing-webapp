# Use the official Python image with a slim version
FROM python:3.10-slim

# Install ffmpeg for moviepy
RUN apt-get update && apt-get install -y ffmpeg

# Set the environment variable to point to the system ffmpeg
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and config file
COPY . .

# Expose the port that Streamlit uses
EXPOSE 8501

# Set the environment variable to point to the config file
ENV STREAMLIT_CONFIG_FILE=/app/config.toml

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
