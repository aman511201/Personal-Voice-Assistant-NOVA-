# ==============================================================================
# NOVA AI Voice Assistant - Production Dockerfile (Hugging Face Spaces Ready)
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install audio and build system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy source code and ensure required directories
COPY --chown=user:user . /home/user/app
RUN mkdir -p /home/user/app/memory /home/user/app/static /home/user/app/audio

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/status || exit 1

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "7860"]
