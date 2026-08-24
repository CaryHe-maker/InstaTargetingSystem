# syntax=docker/dockerfile:1.7

# The official PyTorch image already contains matching CUDA torch and
# torchvision builds. Installing the development requirements would download
# them again and add a second large package layer.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.txt \
    && python -c "import cv2, easydict, numpy, sys, timm, torch, torchvision, yaml; assert sys.version_info[:2] == (3, 12); assert torch.__version__.startswith('2.11.0'); assert torchvision.__version__.startswith('0.26.0'); assert torch.version.cuda == '12.8'; assert 'sm_120' in torch._C._cuda_getArchFlags().split()" \
    && python -m pip uninstall -y torchaudio \
    && rm -rf \
        /root/.cache \
        /tmp/* \
        /usr/share/doc/* \
        /usr/share/man/* \
    && find /usr/local/lib/python3.12/dist-packages -type d \
        -name __pycache__ -prune -exec rm -rf '{}' + \
    && find /usr/local/lib/python3.12/dist-packages -type f \
        \( -name '*.pyc' -o -name '*.pyo' -o -name '*.a' \) -delete

# The devel image also ships the PyTorch source tree, CUDA compiler toolkit,
# profilers, headers, and extension-build helpers. The competition route only
# needs the prebuilt torch and NVIDIA wheel runtime libraries.
RUN rm -rf \
        /opt/nvidia \
        /opt/pytorch \
        /usr/local/cuda \
        /usr/local/cuda-12.8 \
        /usr/local/lib/python3.12/dist-packages/cmake \
        /usr/local/lib/python3.12/dist-packages/cmake-* \
        /usr/local/lib/python3.12/dist-packages/cuda \
        /usr/local/lib/python3.12/dist-packages/cuda_* \
        /usr/local/lib/python3.12/dist-packages/_cuda_bindings_redirector.pth \
        /usr/local/lib/python3.12/dist-packages/_cuda_bindings_redirector.py \
        /usr/local/lib/python3.12/dist-packages/torch/bin/protoc \
        /usr/local/lib/python3.12/dist-packages/torch/bin/protoc-3.13.0.0 \
        /usr/local/lib/python3.12/dist-packages/torch/include \
        /usr/local/lib/python3.12/dist-packages/torch/share \
        /usr/local/lib/python3.12/dist-packages/triton \
        /usr/local/lib/python3.12/dist-packages/triton-* \
    && find /usr/local/lib/python3.12/dist-packages/nvidia -type d \
        -name include -prune -exec rm -rf '{}' + \
    && python -c "import cv2, easydict, numpy, timm, torch, torchvision, yaml; assert torch.version.cuda == '12.8'; assert 'sm_120' in torch._C._cuda_getArchFlags().split()"

WORKDIR /app

COPY src ./src
COPY configs/RGBonly.yaml ./configs/RGBonly.yaml
COPY models/artrackv2_b_256.pth.tar ./models/artrackv2_b_256.pth.tar
COPY track.py ./track.py

# Fail the build if context filtering breaks the production import graph, the
# PostTrainingV1.3 wrapped-BFoV Geometry regression, or the strict checkpoint.
RUN PYTHONPATH=/app/src python -c "import sys; from instatarget.app.competition import runCompetition; from instatarget.core.types import BBoxXYWH; from instatarget.geometry import SphericalGeometryImpl; from instatarget.tracker.artrack_model import PyTorchARTrackV2Session; geometry = SphericalGeometryImpl(boundarySamplesPerEdge=33); wrapped_bfov = geometry.bboxToBfov(BBoxXYWH(xPx=0.0, yPx=6.0, widthPx=174.0, heightPx=126.0), 360, 180); assert 0.0 < wrapped_bfov.horizontalFovRad < 3.141592653589793; assert 0.0 < wrapped_bfov.verticalFovRad < 3.141592653589793; assert 'instatarget.data' not in sys.modules; print('ARTrackV2 runtime imports and wrapped BFoV Geometry verified')"

COPY docker/partition_image.py /partition_image.py
RUN python /partition_image.py --output /layer-parts --layers 7 \
    && rm /partition_image.py

# Rebuild the cleaned filesystem from seven balanced, disjoint buckets. This is
# below the submission limit of ten RootFS layers; ENV and ENTRYPOINT are
# metadata-only instructions and do not add filesystem layers.
FROM scratch
COPY --from=runtime /layer-parts/00/ /
COPY --from=runtime /layer-parts/01/ /
COPY --from=runtime /layer-parts/02/ /
COPY --from=runtime /layer-parts/03/ /
COPY --from=runtime /layer-parts/04/ /
COPY --from=runtime /layer-parts/05/ /
COPY --from=runtime /layer-parts/06/ /

ENV PATH=/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DATASET_DIR=/mnt/dataset \
    RESULT_DIR=/mnt/result \
    CONFIG_PATH=/app/configs/RGBonly.yaml \
    ARTRACKV2_ROOT=/app/src/instatarget/vendor/artrackv2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/track.py"]
