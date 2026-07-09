FROM python:3.9-slim

# Install system dependencies untuk OpenCV & YOLO
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user UID 1000 (Syarat Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /code

# Install dependencies (CPU-only PyTorch)
COPY --chown=user:user requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY --chown=user:user . /code

# Pastikan folder static dan subfoldernya ada dan writable
RUN mkdir -p /code/static && chmod 777 /code/static

# Jalankan Flask melalui Gunicorn di port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "120"]
