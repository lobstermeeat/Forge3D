# syntax=docker/dockerfile:1.7
#
# FORGE 3D reference-image worker: FLUX.1 [schnell] (Apache-2.0) on RunPod serverless.
#
#   docker build --secret id=hf_token,env=HF_TOKEN -t forge3d/flux-schnell-worker workers/flux-schnell
#
# Needs ~34 GB of VRAM (48 GB cards such as L40S, RTX A6000, A40). On 24 GB cards set
# FLUX_CPU_OFFLOAD=1, which works but is several times slower.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLUX_MODEL_DIR=/models/FLUX.1-schnell \
    HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY scripts/download_weights.py /opt/worker/scripts/download_weights.py
RUN --mount=type=secret,id=hf_token \
    HF_HUB_OFFLINE=0 HF_TOKEN="$(cat /run/secrets/hf_token)" python /opt/worker/scripts/download_weights.py

COPY reference_worker /opt/worker/reference_worker
COPY handler.py /opt/worker/handler.py
WORKDIR /opt/worker
CMD ["python", "-u", "handler.py"]
