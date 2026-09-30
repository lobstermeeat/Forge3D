"""Web packaging via gltfpack: meshopt-compressed geometry, WebP colour and KTX2 metallic-roughness."""

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
        # Never ship textures larger than the preset. The global limit goes first: a later one would
        # override the metallic-roughness limit below
        "-tl", str(texture_limit),
        # Colour as WebP (EXT_texture_webp). Basis ETC1S shares one codebook across the atlas, and at
        # the smaller mip levels, which is what a model shows at normal viewing distance, its 4x4
        # blocks mixed neighbouring UV charts into grime and specks. The browser makes WebP's mips.
        "-tw", "color",
        # Metallic-roughness as UASTC (KTX2, KHR_texture_basisu) at a quarter of the size: ETC1S
        # shifted every channel per block, scattering metal and gloss specks over matte surfaces
        "-tu", "attrib",
        "-tl", "attrib", str(max(256, texture_limit // 4)),
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
