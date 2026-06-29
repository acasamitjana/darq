FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --upgrade pip

# Install CPU version of PyTorch first.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY . /app

# Install DARQ as a normal package inside the container.
RUN python -m pip install --no-cache-dir .

CMD ["darq", "--help"]
