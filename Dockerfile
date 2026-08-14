FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs/RGBonly.yaml ./configs/RGBonly.yaml
COPY third_party/HiT ./third_party/HiT
COPY models/hit_small.pth ./models/hit_small.pth
COPY track.py ./track.py

ENV DATASET_DIR=/mnt/dataset \
    RESULT_DIR=/mnt/result \
    CONFIG_PATH=/app/configs/RGBonly.yaml \
    HIT_ROOT=/app/third_party/HiT \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "track.py"]
