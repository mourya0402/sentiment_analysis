FROM python:3.11-slim

# tini for clean PID1; node exporter for system metrics
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini prometheus-node-exporter \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# deps first (better cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY . /app

# Ports:
# 7860 = Gradio UI, 8000 = Python /metrics, 9100 = Node Exporter
EXPOSE 7860 8000 9100

ENTRYPOINT ["/usr/bin/tini", "--"]
# Start Node Exporter in background, then your app
CMD prometheus-node-exporter --web.listen-address=":9100" & python app.py
