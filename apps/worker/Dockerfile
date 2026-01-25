# Stream2Short Worker Dockerfile
FROM python:3.11-slim

# Install system dependencies, OpenCV libs, and fonts for subtitles
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    unzip \
    # OpenCV dependencies for face detection
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
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

# Download OpenCV DNN face detection models (ResNet SSD - much better than Haar)
# These are required for reliable face tracking in FULL_CAM mode
RUN mkdir -p /app/models \
    && wget -q "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt" \
         -O /app/models/deploy.prototxt \
    && wget -q "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel" \
         -O /app/models/res10_300x300_ssd_iter_140000.caffemodel \
    && echo "✅ DNN face detection models downloaded"

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

