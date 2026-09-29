"""Web packaging: meshopt-compressed geometry and KTX2 (Basis Universal) textures via gltfpack."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

GLTFPACK = os.environ.get("GLTFPACK_BIN", "gltfpack")


class CompressionError(RuntimeError):
    pass


def gltfpack_args(texture_limit: int) -> list[str]:
    return [
        "-cc",  # meshopt compression, higher ratio (EXT_meshopt_compression + KHR_mesh_quantization)
        "-tc",  # KTX2 with Basis Universal supercompression (KHR_texture_basisu)
        "-tq", "8",  # texture quality 1-10
        "-tl", str(texture_limit),  # never ship textures larger than the preset
    ]


def pack_glb(glb: bytes, texture_limit: int) -> bytes:
    """Compress a GLB for the browser. The client decodes it with MeshoptDecoder and KTX2Loader."""
    if shutil.which(GLTFPACK) is None:
        raise CompressionError(f"{GLTFPACK} not found; install gltfpack from meshoptimizer releases")
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "in.glb")
        target = os.path.join(tmp, "out.glb")
        with open(source, "wb") as f:
            f.write(glb)
        result = subprocess.run(
            [GLTFPACK, "-i", source, "-o", target, *gltfpack_args(texture_limit)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0 or not os.path.exists(target):
            raise CompressionError(f"gltfpack failed: {result.stderr.strip() or result.stdout.strip()}")
        with open(target, "rb") as f:
            return f.read()
