FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV, PDF processing, and QR code detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories
RUN mkdir -p /app/input /app/intermediate

CMD ["python", "-u", "src/main.py"]
