#!/bin/bash
set -e

FFMPEG_PATH="/workspace/ffmpeg"
FFPROBE_PATH="/workspace/ffprobe"

if [ ! -f "$FFMPEG_PATH" ]; then
    echo "Downloading ffmpeg..."
    curl -L "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz" -o /tmp/ffmpeg.tar.xz
    echo "Extracting ffmpeg..."
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp/
    cp /tmp/ffmpeg-master-latest-linux64-gpl*/bin/ffmpeg $FFMPEG_PATH
    cp /tmp/ffmpeg-master-latest-linux64-gpl*/bin/ffprobe $FFPROBE_PATH
    chmod +x $FFMPEG_PATH $FFPROBE_PATH
    rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-master-latest-linux64-gpl*
    echo "ffmpeg ready."
else
    echo "ffmpeg already exists, skipping download."
fi

export PATH="/workspace:$PATH"
echo "Starting gunicorn..."
exec gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1
