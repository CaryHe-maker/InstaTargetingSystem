# syntax=docker/dockerfile:1.7

# The official PyTorch image already contains matching CUDA torch and
# torchvision builds. Installing the development requirements would download
# them again and add a second large package layer.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.txt \
    && python -c "import cv2, easydict, numpy, timm, torch, torchvision, yaml; assert torch.__version__.startswith('2.6.0'); assert torchvision.__version__.startswith('0.21.0')" \
    && python -m pip uninstall -y torchaudio \
    && conda clean -afy \
    && rm -rf \
        /root/.cache \
        /tmp/* \
        /usr/share/doc/* \
        /usr/share/man/* \
        /opt/conda/include \
        /opt/conda/lib/python3.11/site-packages/torch/include \
        /opt/conda/lib/python3.11/site-packages/torch/share \
        /opt/conda/lib/python3.11/site-packages/torch/test \
        /opt/conda/lib/python3.11/site-packages/torchgen \
    && find /opt/conda -type d \( -name __pycache__ -o -name tests \) -prune -exec rm -rf '{}' + \
    && find /opt/conda -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.a' \) -delete

WORKDIR /app

COPY src ./src
COPY configs/RGBonly.yaml ./configs/RGBonly.yaml
COPY models/hit_small_inference.pth ./models/hit_small.pth
COPY track.py ./track.py

# A scratch final stage keeps only the cleaned filesystem state. Deleted base
# files and package caches no longer remain referenced by lower image layers.
FROM scratch
COPY --from=runtime / /

WORKDIR /app

ENV PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DATASET_DIR=/mnt/dataset \
    RESULT_DIR=/mnt/result \
    CONFIG_PATH=/app/configs/RGBonly.yaml \
    HIT_ROOT=/app/src/instatarget/vendor/hit \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "track.py"]
