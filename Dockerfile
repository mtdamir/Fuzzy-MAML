
# Use official TensorFlow 1.15 CPU image (includes TensorFlow)
FROM tensorflow/tensorflow:1.15.0-py3

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy the entire project (including data folder) into the container
COPY . /app

# Install additional Python packages if needed
RUN pip install --upgrade pip && \
    pip install numpy scipy pillow matplotlib

# Start a bash shell by default so you can run scripts manually
CMD ["/bin/bash"]
