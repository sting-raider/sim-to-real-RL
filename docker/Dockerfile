# SO-100 Sim-to-Real Training Environment
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/sim-to-real-arm

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install IsaacGym (requires manual download from NVIDIA first)
# COPY isaacgym.tar.gz /tmp/
# RUN cd /tmp && tar -xzf isaacgym.tar.gz && \
#     cd isaacgym/python && pip3 install -e .

# Copy project
COPY . .

# Default command
CMD ["python3", "sim/train.py", "--help"]
