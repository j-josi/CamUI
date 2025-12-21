# ---------------------------------------------
# Base-Image
# ---------------------------------------------
FROM dtcooper/raspberrypi-os:bookworm
LABEL maintainer="souri-t"

# ---------------------------------------------
# Add raspberry pi repositories (with certificates)
# ---------------------------------------------
RUN apt update -y \
    && apt install -y --no-install-recommends curl gnupg ca-certificates \
    && update-ca-certificates \
    && curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/raspberrypi.gpg \
    && echo "deb http://archive.raspberrypi.org/debian bookworm main" \
        > /etc/apt/sources.list.d/raspi.list \
    && apt update -y

# ---------------------------------------------
# Install apt pakages (git, vim, libcamera, picamera2 and ffmpeg)
# ---------------------------------------------
RUN apt upgrade -y \
    && apt install -y --no-install-recommends \
        git \
        vim \
        python3-pip \
        python3-picamera2 \
        libcamera-apps \
        libcamera-tools \
        libcap-dev \
        libcamera-dev \
        libatlas-base-dev \
        libopenjp2-7 \
        libkms++-dev \
        libfmt-dev \
        libdrm-dev \
        ffmpeg \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------
# Set work dir
# ---------------------------------------------
WORKDIR /app

# ---------------------------------------------
# Copy Code of CamUI application
# ---------------------------------------------
COPY static /app/static
COPY templates /app/templates
COPY requirements.txt /app/
COPY *.py /app/
COPY *.json /app/

# ---------------------------------------------
# Install pip pakages
# ---------------------------------------------
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# ---------------------------------------------
# Run/start gunicorn on container start
# ---------------------------------------------
CMD ["gunicorn", \
     "app:app", \
     "--workers", "1", \
     "--threads", "8", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "0"]
