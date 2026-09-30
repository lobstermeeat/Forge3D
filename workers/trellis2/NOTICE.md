# Third-party notices: TRELLIS.2 worker

**Built with DINOv3.**

Orainge's AI generation runs the components below. Anything that shows AI-generated
assets to users (for example the editor's AI panel) must display "Built with DINOv3",
which the DINOv3 License requires.

| Component                                                                                                                                                                   | Used for                                                  | License                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) code and [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) weights                                          | Image to 3D                                               | MIT                                                                                 |
| [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) sparse-structure decoder                                                                        | Used by TRELLIS.2                                         | MIT                                                                                 |
| [o-voxel](https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel), [CuMesh](https://github.com/JeffreyXiang/CuMesh), [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) | Mesh extraction, remeshing, UV unwrap, sparse convolution | MIT                                                                                 |
| [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)                                                                                          | Image encoder                                             | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license) |
| [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)                                                                                                                      | Background removal                                        | MIT                                                                                 |
| [flash-attention](https://github.com/Dao-AILab/flash-attention)                                                                                                             | Attention kernels                                         | BSD-3-Clause                                                                        |
| [PyTorch](https://github.com/pytorch/pytorch)                                                                                                                               | Runtime                                                   | BSD-3-Clause                                                                        |
| [gltfpack / meshoptimizer](https://github.com/zeux/meshoptimizer) (embeds [Basis Universal](https://github.com/BinomialLLC/basis_universal))                                | meshopt + KTX2 packing                                    | MIT (Basis Universal: Apache-2.0)                                                   |

## Deliberately excluded

| Component                                                                                      | Why                                                                                                               | Replacement                                                                                        |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast), nvdiffrec                                  | NVIDIA Source Code License: research or evaluation use only                                                       | `forge3d_worker/uv_raster.py` implements the UV-space rasterize/interpolate the texture bake needs |
| [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) (stock TRELLIS.2 background removal) | CC BY-NC 4.0: no commercial use without a BRIA agreement                                                          | BiRefNet (MIT)                                                                                     |
| Tencent Hunyuan3D 2.1                                                                          | Its license does not apply in South Korea, the EU or the UK, and forbids using its output to improve other models | TRELLIS.2                                                                                          |

This summary is not legal advice; have counsel confirm it before launch.
