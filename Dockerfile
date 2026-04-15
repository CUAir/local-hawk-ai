FROM python:3.9

WORKDIR /app

# Install system dependencies for OpenCV and Detectron2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Entry point to intsys client app
ENTRYPOINT ["python", "core.py"]
