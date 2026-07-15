FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DARQ_SERVER_NAME=0.0.0.0
ENV DARQ_SERVER_PORT=7860

WORKDIR /app

RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch

COPY . /app

# Install DARQ and the remaining project dependencies.
RUN python -m pip install --no-cache-dir .

CMD ["darq", "--help"]