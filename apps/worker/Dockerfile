# Stream2Short Worker Dockerfile
FROM python:3.11-slim

# Install system dependencies and fonts for subtitles
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    unzip \
    # Fonts for social media style subtitles
    fonts-liberation \
    fonts-dejavu-core \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Download and install Montserrat font (Google Fonts - recommended for TikTok/Reels style)
RUN mkdir -p /usr/share/fonts/truetype/montserrat \
    && wget -q "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf" -O /usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf \
    && wget -q "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-SemiBold.ttf" -O /usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf \
    && wget -q "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf" -O /usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf \
    && fc-cache -f -v

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p /tmp/stream2short

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TEMP_DIR=/tmp/stream2short

# Run the worker
CMD ["python", "main.py"]

