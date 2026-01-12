# Base image for Jetson Orin with JetPack 6.x
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

WORKDIR /app

# Install poppler-utils for PDF to image conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install EasyOCR with compatible versions
# Use opencv-python-headless 4.9.x which supports numpy 1.x
RUN pip3 install --no-cache-dir \
    python-bidi pyclipper Shapely "scikit-image<0.24" imageio tifffile lazy-loader \
    "opencv-python-headless==4.9.0.80" && \
    pip3 install --no-cache-dir easyocr --no-deps && \
    pip3 install --no-cache-dir "numpy==1.26.1" --force-reinstall

# Fix protobuf to be compatible with TensorFlow 2.14
RUN pip3 uninstall -y protobuf && \
    pip3 install --no-cache-dir "protobuf==3.20.3"

# Copy source code
COPY src/ ./src/

# Create directories
RUN mkdir -p /app/intermediate /app/output /app/templates/debug_cells /app/training_data /app/models

# Set environment for GPU
ENV CUDA_VISIBLE_DEVICES=0

# Expose scanner server port
EXPOSE 8080

# Start script that runs either scanner server or task watcher
COPY src/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
