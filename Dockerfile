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
    && find /opt/conda -type d \( -name __pycache__ -o -name tests \) -prune -exec rm -rf '{}' + \
    && find /opt/conda -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.a' \) -delete

WORKDIR /app

COPY src ./src
COPY configs/RGBonly.yaml ./configs/RGBonly.yaml
COPY models/hit_small_inference.pth ./models/hit_small.pth
COPY track.py ./track.py

COPY docker/partition_image.py /partition_image.py
RUN python /partition_image.py --output /layer-parts --layers 50 \
    && rm /partition_image.py

# Rebuild the cleaned filesystem from 50 disjoint buckets. Large CUDA/PyTorch
# shared libraries are assigned to their own buckets; all other files are
# balanced over the remaining buckets. This keeps registry blobs bounded while
# preserving the same final filesystem.
FROM scratch
COPY --from=runtime /layer-parts/00/ /
COPY --from=runtime /layer-parts/01/ /
COPY --from=runtime /layer-parts/02/ /
COPY --from=runtime /layer-parts/03/ /
COPY --from=runtime /layer-parts/04/ /
COPY --from=runtime /layer-parts/05/ /
COPY --from=runtime /layer-parts/06/ /
COPY --from=runtime /layer-parts/07/ /
COPY --from=runtime /layer-parts/08/ /
COPY --from=runtime /layer-parts/09/ /
COPY --from=runtime /layer-parts/10/ /
COPY --from=runtime /layer-parts/11/ /
COPY --from=runtime /layer-parts/12/ /
COPY --from=runtime /layer-parts/13/ /
COPY --from=runtime /layer-parts/14/ /
COPY --from=runtime /layer-parts/15/ /
COPY --from=runtime /layer-parts/16/ /
COPY --from=runtime /layer-parts/17/ /
COPY --from=runtime /layer-parts/18/ /
COPY --from=runtime /layer-parts/19/ /
COPY --from=runtime /layer-parts/20/ /
COPY --from=runtime /layer-parts/21/ /
COPY --from=runtime /layer-parts/22/ /
COPY --from=runtime /layer-parts/23/ /
COPY --from=runtime /layer-parts/24/ /
COPY --from=runtime /layer-parts/25/ /
COPY --from=runtime /layer-parts/26/ /
COPY --from=runtime /layer-parts/27/ /
COPY --from=runtime /layer-parts/28/ /
COPY --from=runtime /layer-parts/29/ /
COPY --from=runtime /layer-parts/30/ /
COPY --from=runtime /layer-parts/31/ /
COPY --from=runtime /layer-parts/32/ /
COPY --from=runtime /layer-parts/33/ /
COPY --from=runtime /layer-parts/34/ /
COPY --from=runtime /layer-parts/35/ /
COPY --from=runtime /layer-parts/36/ /
COPY --from=runtime /layer-parts/37/ /
COPY --from=runtime /layer-parts/38/ /
COPY --from=runtime /layer-parts/39/ /
COPY --from=runtime /layer-parts/40/ /
COPY --from=runtime /layer-parts/41/ /
COPY --from=runtime /layer-parts/42/ /
COPY --from=runtime /layer-parts/43/ /
COPY --from=runtime /layer-parts/44/ /
COPY --from=runtime /layer-parts/45/ /
COPY --from=runtime /layer-parts/46/ /
COPY --from=runtime /layer-parts/47/ /
COPY --from=runtime /layer-parts/48/ /
COPY --from=runtime /layer-parts/49/ /

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

ENTRYPOINT ["python", "/app/track.py"]
