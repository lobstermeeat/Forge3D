# MV-Adapter (vendored subset)

The image-to-multiview pipeline of [MV-Adapter](https://github.com/huanngzh/MV-Adapter) by Zehuan
Huang et al., Apache License 2.0 (`LICENSE` in this folder), copied from commit
`4277e0018232bac82bb2c103caf0893cedb711be` (26 Jun 2025, upstream `main` on 1 Oct 2026). The
pipeline file also carries the HuggingFace Team's Apache-2.0 header, as upstream.

Only what `scripts/inference_i2mv_sdxl.py` needs to run without a mesh is here, under the same
paths as upstream so its imports work unchanged:

| File                                       | Status                                                      |
| ------------------------------------------ | ----------------------------------------------------------- |
| `__init__.py`, `models/__init__.py`        | unchanged (empty)                                           |
| `loaders/__init__.py`, `loaders/custom_adapter.py` | unchanged                                           |
| `models/attention_processor.py`            | unchanged                                                   |
| `pipelines/pipeline_mvadapter_i2mv_sdxl.py` | unchanged                                                  |
| `schedulers/scheduler_utils.py`, `schedulers/scheduling_shift_snr.py` | unchanged                         |
| `utils/geometry.py`                        | unchanged                                                   |
| `pipelines/__init__.py`, `schedulers/__init__.py` | added, empty (upstream relies on namespace packages) |
| `utils/__init__.py`                        | **modified**: imports only the camera and Plücker helpers   |
| `utils/mesh_utils/__init__.py`             | **modified**: exports only the camera helpers               |
| `utils/mesh_utils/camera.py`               | **modified**: `LIST_TYPE` defined locally; unused `trimesh` and `PIL` imports dropped |

Not included: the text-to-multiview, SD2.1, geometry-guided and texturing pipelines, training code,
and the rest of `utils/` and `utils/mesh_utils/` (rendering, projection, UV and painting), which
import nvdiffrast. nvdiffrast's license allows research use only, so it is never installed or
imported (`../tests/test_cameras.py` checks that the vendored code loads without it).

The adapter weights (`huanngzh/mv-adapter`, `mvadapter_i2mv_sdxl.safetensors`) are Apache-2.0 too;
they adapt Stable Diffusion XL 1.0 (CreativeML Open RAIL++-M) and use the SDXL-VAE-FP16-Fix decoder
(MIT). See `../scripts/download_weights.py` for the pinned revisions.
