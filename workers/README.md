# AI workers

FORGE 3D generates 3D models with its own models on RunPod serverless GPUs. There are no
third-party AI APIs involved.

| Worker | Model | Job | GPU |
| --- | --- | --- | --- |
| [`trellis2/`](trellis2) | TRELLIS.2-4B (MIT) | image to textured GLB | 24 GB+ Ada/Ampere/Hopper (RTX 4090, L40S, A6000, H100) |
| [`flux-schnell/`](flux-schnell) | FLUX.1 [schnell] (Apache-2.0) | text to reference images | 48 GB (L40S, A6000, A40) |

The server talks to both through `apps/server/src/services/ai` (`SelfHostedProvider`).

## The flow: spend GPU time only on results people keep

1. **Text prompt → 4 reference images** (FLUX.1 [schnell], a few seconds). The prompt is wrapped
   in a product-shot template: one centred object, plain background, soft light, 3/4 view from
   slightly above. That is the input image-to-3D handles best.
2. **User picks one → preview** (`mode: "preview"`): TRELLIS.2 at 512³, 30k triangles, 1K
   textures.
3. **User keeps it → final** (`mode: "final"`, *same seed*): 1024³ cascade, 100k triangles,
   2K textures. The same seed gives the same coarse structure, so the final refines the preview
   the user approved.
4. Both passes are packed for the browser with gltfpack: meshopt geometry and KTX2 (Basis
   Universal) textures. The editor's `AssetLoader` decodes them; the decoders are served from
   `/decoders/` by the client's Vite config.

Uploaded images can skip step 1. Images with transparency skip background removal.

## Job contracts

`trellis2` input:

```json
{ "image_url": "https://…", "mode": "preview", "seed": 1234, "request_id": "gen_42" }
```

`image_base64` can replace `image_url`. `seed` is optional (a random one is returned) and
`request_id` names the output (`ai/<request_id>/<mode>.glb`). Output:

```json
{ "request_id": "gen_42", "mode": "preview", "seed": 1234,
  "glb": { "key": "ai/gen_42/preview.glb", "url": "https://assets…/ai/gen_42/preview.glb" },
  "bytes": 812345, "raw_bytes": 3012345, "triangles": 30000,
  "timings": { "generate_s": 9.8, "export_s": 4.1, "compress_s": 2.2, "upload_s": 0.3 },
  "credits": ["Built with DINOv3", "3D generation: TRELLIS.2 (Microsoft, MIT)"] }
```

`flux-schnell` input: `{ "prompt": "a brass pocket watch", "count": 4, "seed": 5, "request_id": "gen_42" }`.
Output: `{ "images": [{ "key", "url", "seed" }, …], "prompt", "seconds" }`.

Invalid input comes back as `{ "error": "invalid input: …" }`. A GPU failure comes back as an
error with `refresh_worker: true`, so RunPod restarts the worker process.

## Deploying

1. On Hugging Face, request access to
   [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (Meta approves it
   manually) and accept the [FLUX.1 [schnell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
   terms. Create a read token.
2. Build and push the images. The token is a build secret, so it never lands in a layer:

   ```sh
   export HF_TOKEN=hf_…
   docker build --secret id=hf_token,env=HF_TOKEN -t <registry>/forge3d-trellis2:1 workers/trellis2
   docker build --secret id=hf_token,env=HF_TOKEN -t <registry>/forge3d-flux-schnell:1 workers/flux-schnell
   docker push <registry>/forge3d-trellis2:1 && docker push <registry>/forge3d-flux-schnell:1
   ```

   Weights are baked in (about 16 GB for TRELLIS.2 and 33 GB for FLUX), so cold starts don't
   download anything. Revisions of every model and repository are pinned.
3. Create one RunPod serverless endpoint per image with these environment variables:

   | Variable | Purpose |
   | --- | --- |
   | `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` | Where outputs go |
   | `R2_PUBLIC_BASE_URL` | Public URL of the bucket, returned to the server |
   | `ALLOWED_IMAGE_HOSTS` (trellis2) | Comma-separated hosts `image_url` may point to, e.g. your R2 domain |
   | `TRELLIS2_LOW_VRAM=0` (trellis2, optional) | Keep all models on the GPU; faster on 48 GB+ cards |
   | `FLUX_CPU_OFFLOAD=1` (flux-schnell, optional) | Run on 24 GB cards, several times slower |

   Without R2 the workers return files inline (base64, 8 MB limit), which is fine for local tests only.
4. Give the server `RUNPOD_API_KEY`, `RUNPOD_TRELLIS2_ENDPOINT_ID` and
   `RUNPOD_REFERENCE_ENDPOINT_ID` (see `.env.example`). `createAIOrchestrator()` registers the
   provider when they are set.

Keep endpoints at **0 minimum workers** (scale to zero). An always-on RTX 4090 pod ($0.74/h)
only beats serverless ($1.10/h, billed while busy) once the GPU would be busy about two-thirds
of the time.

### RTX 5090 and other Blackwell cards

The image targets CUDA 12.4 (compute capability 8.6, 8.9 and 9.0). Blackwell (sm_120) needs
CUDA 12.8+: switch the base image to `nvidia/cuda:12.8.x-cudnn-devel-ubuntu22.04`, install a
cu128 build of PyTorch, add `12.0` to `TORCH_CUDA_ARCH_LIST`, and use a flash-attn build for
that torch version. CuMesh, FlexGEMM and o-voxel must be compiled against that exact PyTorch,
which is what the Dockerfile does; a mismatch shows up as an ABI error at import.

## Cost (estimates to check against real `timings`)

RunPod list prices on 29 Sep 2026: RTX 4090 $1.10/h serverless ($0.74/h always-on), RTX 5090
$1.58/h ($0.99/h), L40S $1.75/h, RTX A6000/A40 $1.22/h.

| Step | GPU time (estimate) | Cost |
| --- | --- | --- |
| 4 reference images, L40S | 10–20 s | ~$0.01 |
| Preview, RTX 4090 | 15–30 s | ~$0.005–0.01 |
| Final, RTX 4090 | 1–2 min | ~$0.02–0.04 |

Microsoft publishes H100 timings only (about 3 s at 512³, 17 s at 1024³ and 60 s at 1536³,
before export). Cold starts add model-loading time on the first job after scaling from zero.

## Licenses

See [`trellis2/NOTICE.md`](trellis2/NOTICE.md). In short: every component is MIT, BSD or
Apache-2.0 except DINOv3, whose license requires showing **"Built with DINOv3"** (it is in the
editor's File › About dialog, and each result carries `credits` to show next to generated
models). Excluded on purpose: nvdiffrast/nvdiffrec (research-only; replaced by
`trellis2/forge3d_worker/uv_raster.py`), RMBG-2.0 (non-commercial; replaced by BiRefNet),
FLUX.1 [dev] (non-commercial) and Hunyuan3D 2.1 (not licensed in South Korea, the EU or the UK).

## Tests

```sh
cd workers/trellis2 && python -m pytest tests      # CPU only; set GLTFPACK_BIN to include gltfpack
cd workers/flux-schnell && python -m pytest tests
```

The tests cover input validation, job handling, the checkpoint check and the nvdiffrast
stand-in (against a brute-force rasterizer and analytic results). Before the first production
deploy, run `workers/trellis2/scripts/compare_nvdiffrast.py` once on a GPU machine that has
nvdiffrast installed (evaluation use) to confirm the stand-in matches it on real hardware.
