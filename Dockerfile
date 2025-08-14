FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv python3-dev git \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*
    
## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
## Include the following line if you have a requirements.txt file.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
