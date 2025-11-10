# Small, stable Python base
FROM python:3.11-slim

# System utilities (optional, but helps with clean signal handling)
RUN apt-get update && apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install Python deps first for better layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . /app

# Gradio will listen on this port inside the container
EXPOSE 7860

# Use tini as PID 1
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run your local product (app.py already binds to 0.0.0.0:7860)
CMD ["python", "app.py"]
