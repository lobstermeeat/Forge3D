# Third-party notices: Pixal3D worker

**Built with DINOv3.**

The Pixal3D worker makes Orainge's finals on Modal, in the TRELLIS.2 worker's container
(`../modal_app.py`; see "The recipe" in `../README.md`). It runs the components below, on top of
everything the TRELLIS.2 worker runs (see `../trellis2/NOTICE.md`; its export, texture bake stand-in
and packing are reused as they are). Both of Pixal3D's weight sets are used: the multi-view set for
most pictures, the single-view set for thin, flat objects.

| Component | Pinned at | Used for | License |
| --- | --- | --- | --- |
| [Pixal3D](https://github.com/TencentARC/Pixal3D) code | commit `f7cf38429b0bd264f1995f0f8743a88b1c728b94` (2026-09-01) | Pixel-aligned image (and multi-view) to 3D | MIT (`LICENSE-Pixal3D.txt`, the repository's LICENSE since commit `5098ba1`, 2026-05-21) |
| [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) weights | revision `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f` (2026-08-31) | Flow models, single-view and multi-view sets | MIT (the model repository's LICENSE, the same text) |
| TRELLIS.2 decoders | the TRELLIS.2 worker's files | Pixal3D's decoders are byte-identical (same sha256) | MIT |
| [MoGe](https://github.com/microsoft/MoGe) code | commit `07444410f1e33f402353b99d6ccd26bd31e469e8` (MoGe-2) | A single picture's field of view | MIT |
| [Ruicheng/moge-2-vitl](https://huggingface.co/Ruicheng/moge-2-vitl) weights | revision `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` | The same | MIT |
| [utils3d](https://github.com/EasternJournalist/utils3d) | commit `3fab839f0be9931dac7c8488eb0e1600c236e183` | MoGe's geometry helpers | MIT |
| [NAF](https://github.com/valeoai/NAF) code and weights | commit `37f2dfc180f2de53d98bd601109c0da0dd6b0f43`; `naf_release.pth` sha256 `c096c1ab…c98f` | Upsamples DINOv3 features for the shape and texture stages | Apache-2.0 |
| DINOv3 ViT-L/16 | the TRELLIS.2 worker's copy (identical to the one Pixal3D names, sha256 checked) | Image features | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license) |
| BiRefNet | the TRELLIS.2 worker's copy | Background removal | MIT |

## Deliberately excluded

| Component | Why | Replacement |
| --- | --- | --- |
| [NATTEN](https://github.com/SHI-Labs/NATTEN) (MIT) | Not a licensing matter: its wheels need PyTorch 2.7+, and the wheel Pixal3D's demo installs only has Hopper (sm_90) kernels | `pixal3d_worker/neighborhood.py`, the one call NAF makes, in PyTorch (tested against NATTEN's own definition) |
| briaai/RMBG-2.0, Pixal3D's stock background remover | CC BY-NC 4.0 | BiRefNet (MIT) |
| nvdiffrast, which Pixal3D's demo installs | Research-only license | The TRELLIS.2 worker's `uv_raster.py` |
| camenduru/dinov3-vitl16-pretrain-lvd1689m, the DINOv3 mirror Pixal3D names | Meta's own repository is the source of record | facebook/dinov3-vitl16-pretrain-lvd1689m (identical file) |

The model repository's card carries `extra_gated_eu_disallowed: true`, but the repository is not
gated and neither license text restricts territory. This summary is not legal advice; have counsel
confirm it before launch.
