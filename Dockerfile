FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Configuración básica
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

# Copiar e instalar requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
