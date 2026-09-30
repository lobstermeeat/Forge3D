"""RunPod serverless entry point for FORGE 3D's TRELLIS.2 worker."""

import runpod

from forge3d_worker.compress import pack_glb
from forge3d_worker.pipeline import Trellis2Runtime
from forge3d_worker.service import handle_job
from forge3d_worker.storage import storage_from_env

STORAGE = storage_from_env()
# Loads the weights during the cold start, before the worker accepts jobs
RUNTIME = Trellis2Runtime()


def handler(job: dict) -> dict:
    return handle_job(job, RUNTIME, STORAGE, pack_glb)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
