# syntax=docker/dockerfile:1.7
#
# FORGE 3D image-to-3D worker: stock TRELLIS.2-4B on RunPod serverless.
#
#   docker build --secret id=hf_token,env=HF_TOKEN -t forge3d/trellis2-worker workers/trellis2
#
# Targets Ampere, Ada and Hopper GPUs (A40, RTX A6000, RTX 4090, L40S, H100) with the CUDA 12.4
# stack TRELLIS.2 is tested on. Blackwell cards (RTX 5090) need CUDA 12.8 and a rebuild: see README.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    MAX_JOBS=8 \
    ATTN_BACKEND=flash_attn \
    TRELLIS2_MODEL_DIR=/models/TRELLIS.2-4B \
    HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git curl unzip ca-certificates \
        libgl1 libglib2.0-0 libjpeg-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# Prebuilt flash-attn for torch 2.6 / CUDA 12 / CPython 3.10 (building it takes over an hour)
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# TRELLIS.2 and its CUDA extensions (all MIT), pinned. nvdiffrast and nvdiffrec are deliberately
# not installed: their licenses allow research use only (see forge3d_worker/uv_raster.py).
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e
RUN git clone https://github.com/microsoft/TRELLIS.2 /opt/trellis2 \
    && git -C /opt/trellis2 checkout ${TRELLIS2_COMMIT} \
    && git -C /opt/trellis2 submodule update --init --recursive \
    && git clone https://github.com/JeffreyXiang/CuMesh /tmp/CuMesh \
    && git -C /tmp/CuMesh checkout ${CUMESH_COMMIT} \
    && git -C /tmp/CuMesh submodule update --init --recursive \
    && pip install /tmp/CuMesh --no-build-isolation \
    && git clone https://github.com/JeffreyXiang/FlexGEMM /tmp/FlexGEMM \
    && git -C /tmp/FlexGEMM checkout ${FLEXGEMM_COMMIT} \
    && git -C /tmp/FlexGEMM submodule update --init --recursive \
    && pip install /tmp/FlexGEMM --no-build-isolation \
    && pip install /opt/trellis2/o-voxel --no-build-isolation \
    && rm -rf /tmp/CuMesh /tmp/FlexGEMM
ENV PYTHONPATH=/opt/trellis2

# gltfpack (meshoptimizer, MIT; embeds Basis Universal, Apache-2.0) for meshopt + KTX2 packing
ARG GLTFPACK_VERSION=1.3
RUN curl -fsSL -o /tmp/gltfpack.zip \
        https://github.com/zeux/meshoptimizer/releases/download/v${GLTFPACK_VERSION}/gltfpack-ubuntu.zip \
    && unzip /tmp/gltfpack.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/gltfpack \
    && rm /tmp/gltfpack.zip

# Bake the weights in so cold starts don't download ~16 GB. DINOv3 is gated on Hugging Face, so
# the build needs a token from an account Meta has approved. The token never lands in a layer.
COPY scripts/download_weights.py /opt/worker/scripts/download_weights.py
RUN --mount=type=secret,id=hf_token \
    HF_HUB_OFFLINE=0 HF_TOKEN="$(cat /run/secrets/hf_token)" python /opt/worker/scripts/download_weights.py

COPY NOTICE.md /opt/worker/NOTICE.md
COPY forge3d_worker /opt/worker/forge3d_worker
COPY handler.py /opt/worker/handler.py
WORKDIR /opt/worker
CMD ["python", "-u", "handler.py"]
