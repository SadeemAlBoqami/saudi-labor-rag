FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install \
    torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install -r requirements.txt

COPY . .

CMD ["python3", "-m", "evaluation.evaluate_rag"]