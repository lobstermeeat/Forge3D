"""RunPod serverless entry point for Orainge's reference-image worker (FLUX.1 [schnell])."""

import os

import runpod

from reference_worker.service import handle_job, make_flux_generator
from reference_worker.storage import storage_from_env

STORAGE = storage_from_env()
# Loads the weights during the cold start, before the worker accepts jobs
GENERATE = make_flux_generator(
    os.environ.get("FLUX_MODEL_DIR", "/models/FLUX.1-schnell"),
    cpu_offload=os.environ.get("FLUX_CPU_OFFLOAD", "0") == "1",
)


def handler(job: dict) -> dict:
    return handle_job(job, GENERATE, STORAGE)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
