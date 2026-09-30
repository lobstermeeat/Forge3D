# AI workers

FORGE 3D generates 3D models with its own models on serverless GPUs, billed only while they
work. There are no third-party AI APIs involved.

| Worker                          | Model                         | Job                      | GPU                                              |
| ------------------------------- | ----------------------------- | ------------------------ | ------------------------------------------------ |
| [`trellis2/`](trellis2)         | TRELLIS.2-4B (MIT)            | image to textured GLB    | 24 GB+ Ampere/Ada/Hopper (L40S, RTX 4090, A100…) |
| [`flux-schnell/`](flux-schnell) | FLUX.1 [schnell] (Apache-2.0) | text to reference images | 48 GB (L40S, A6000, A40)                         |

They run on [Modal](https://modal.com) (`modal_app.py`, the simplest way to start) or on
RunPod serverless (the Dockerfiles). Both hosts speak the same job protocol, so the server talks
to either through `apps/server/src/services/ai` (`SelfHostedProvider`).

## The flow: spend GPU time only on results people keep

1. **Text prompt → 4 reference images** (FLUX.1 [schnell], a few seconds). The prompt is wrapped
   in a product-shot template: one centred object, plain background, soft light, 3/4 view from
   slightly above. That is the input image-to-3D handles best.
2. **User picks one → preview** (`mode: "preview"`): TRELLIS.2 at 512³, 30k triangles, 1K
   textures.
3. **User keeps it → final** (`mode: "final"`, _same seed_): 1024³ cascade, 100k triangles,
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

`image_base64` can replace `image_url`, which is only fetched from hosts listed in
`ALLOWED_IMAGE_HOSTS`. `seed` is optional (a random one is returned). Outputs are stored at
`ai/<request_id>/<mode>-<seed>.glb`, so every result has its own URL and caches never serve a
stale one. Output:

```json
{
  "request_id": "gen_42",
  "mode": "preview",
  "seed": 1234,
  "glb": {
    "key": "ai/gen_42/preview-1234.glb",
    "url": "https://assets…/ai/gen_42/preview-1234.glb"
  },
  "bytes": 812345,
  "raw_bytes": 3012345,
  "triangles": 30000,
  "timings": { "generate_s": 9.8, "export_s": 4.1, "compress_s": 2.2, "upload_s": 0.3 },
  "credits": ["Built with DINOv3", "3D generation: TRELLIS.2 (Microsoft, MIT)"]
}
```

`flux-schnell` input: `{ "prompt": "a brass pocket watch", "count": 4, "seed": 5, "request_id": "gen_42" }`.
Output: `{ "images": [{ "key", "url", "seed" }, …], "prompt", "seconds" }`.

Invalid input (including an image where no object stands out from the background) comes back
as `{ "error": "invalid input: …" }`. Other failures come back as `generation failed: …`. After
a GPU fault the worker also replaces its container (RunPod: `refresh_worker`; Modal: the
container stops taking jobs), because CUDA may be unusable in that process.

Without R2 (below) files come back inline as `base64` instead of `url` (8 MB limit), and the
server passes them around as `data:` URLs. That is fine for trying things out.

## Deploying on Modal

Modal bills GPUs by the second and includes $30 of free compute a month on its Starter plan.
`modal_app.py` defines both workers, a volume for the weights and a small job API
(`job_api.py`) with the same routes as a RunPod endpoint (`/run`, `/runsync`, `/status/{id}`,
`/cancel/{id}`), under `/trellis2` and `/reference`.

1. On Hugging Face, request access to
   [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (Meta approves it
   manually, so do this first) and accept the
   [FLUX.1 [schnell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell) terms. Create a
   read token.
2. Install the CLI and log in (from the repository root; the same commands work in PowerShell):

   ```sh
   pip install modal
   modal setup
   ```

3. Store two secrets. The worker token is a password you make up for the job API; the server
   sends it with every request:

   ```sh
   modal secret create huggingface-secret HF_TOKEN=hf_…
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   modal secret create orainge-worker-token ORAINGE_WORKER_TOKEN=<the printed value>
   ```

   (You can also create them in the Modal dashboard under Secrets; its Hugging Face template
   makes `huggingface-secret` with the key `HF_TOKEN`.)

4. Build and deploy. This is the only step that needs this computer online: it takes 20–40
   minutes the first time (the TRELLIS.2 image compiles CUDA extensions). If the connection
   drops, run it again; finished build steps are kept.

   ```sh
   modal deploy workers/modal_app.py
   ```

   Give the server the URL it prints (`https://<workspace>--orainge-ai-api.modal.run`) as
   `AI_WORKERS_URL` and the same token as `AI_WORKERS_TOKEN` (see `.env.example`).

5. Make a model. This runs entirely in Modal's cloud, and with `--detach` it keeps going if
   this computer sleeps or goes offline:

   ```sh
   modal run --detach workers/modal_app.py::make --prompt "a brass pocket watch" --final
   ```

   The first run downloads the weights (about 50 GB, 10–30 minutes) into the `orainge-models`
   volume; later runs start within a couple of minutes. Each step is saved in the
   `orainge-outputs` volume under the run's name: the reference images, `preview-<seed>.glb`,
   `final-<seed>.glb` and `progress.json`. If you are still connected at the end, they are also
   copied to `orainge-outputs/<run>/` here.

   | To…                                  | Run                                                                          |
   | ------------------------------------ | ---------------------------------------------------------------------------- |
   | See what's ready, running or failed  | `python workers/modal_app.py status`                                         |
   | Continue an unfinished run           | `modal run --detach workers/modal_app.py::make --run <name>`                 |
   | Make the final of a previewed run    | the same, with `--final`                                                     |
   | Start from your own image            | `make --image photo.png` instead of `--prompt`                               |
   | Make a whole test set                | `modal run --detach workers/modal_app.py::make_set --prompts <file>`         |
   | Download a run                       | `modal volume get orainge-outputs <name> .`                                  |
   | Fetch FLUX while Meta reviews DINOv3 | `modal run --detach workers/modal_app.py::download_models --which reference` |

   A continued run skips every finished step, so finished work is never paid for twice. The GLBs are
   meshopt/KTX2-compressed: open them in the Orainge editor (File › Import model) or another
   viewer that supports those extensions.

**Without your computer:** the `AI ops (Modal)` GitHub workflow runs the same commands on
GitHub's servers. Add the repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (Modal >
Settings > API Tokens), put the commands in `ops/run.sh` and push it to the `ai-ops` branch.
The log and anything the script writes to `ops-out/` land on the `ai-results` branch under
`runs/<id>/`, with the tokens scrubbed. In a public repository that branch is public, so what
the script writes to `ops-out/private/` is saved only encrypted, for the holder of the private
key matching `ops/results-public-key.pem` (`ops/seal.sh` explains how to make one and open the
results). `bash ops/gallery.sh "<title>"` puts a test set's gallery and final models there.

**Review a test set:** a set is a text file with one prompt, or the path of a photo, per line;
`workers/test-sets/starter.txt` covers eight kinds of game assets, and `renders.txt` six photos
of CC0 sample models (no FLUX needed). `make_set` runs every line
(previews and finals; `--no-final` for previews only) with the GPU containers kept warm between
runs, names the runs `<set>-<nn>-<words>` and copies them to `orainge-outputs/<set>/`. Run it
again with the same `--name` to retry what failed. Then
`python workers/gallery/make_gallery.py orainge-outputs/<set> -o gallery/` renders every preview and
final from six angles, with the editor's decoders and lighting, and writes a page showing each
run's reference images, triangles, file size, GPU time and cost, with a 3D viewer (orbit,
zoom, wireframe) for each final. It needs `pnpm install` and Playwright's Chromium, which
`ops/gallery.sh` installs on GitHub's servers; serve the page over http(s) to use the viewer.

**Settings** (in `modal_app.py`): both workers run on an L40S (48 GB), scale to zero, stay warm
for 60 s after their last job (idle time is billed; a cold start takes about a minute) and are
capped at 2 TRELLIS.2 containers and 1 FLUX container to bound spending. `TRELLIS2_GPU = "A10"`
costs about half as much per second but is slower and has only 24 GB; set
`TRELLIS2_LOW_VRAM = "1"` with it.

**R2 storage** (for production): put the storage variables from the RunPod table below in a
Modal secret, then deploy with its name in `ORAINGE_R2_SECRET`:

```sh
modal secret create orainge-r2 R2_ACCOUNT_ID=… R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… R2_BUCKET=… R2_PUBLIC_BASE_URL=https://assets.example.com ALLOWED_IMAGE_HOSTS=assets.example.com
ORAINGE_R2_SECRET=orainge-r2 modal deploy workers/modal_app.py
```

In PowerShell the second line is `$env:ORAINGE_R2_SECRET = "orainge-r2"; modal deploy workers/modal_app.py`.
`modal deploy` prints which storage the workers use, so a deploy without R2 doesn't go unnoticed.

## Deploying on RunPod

1. Hugging Face access as in Modal step 1.
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

   | Variable                                                                 | Purpose                                                                                                          |
   | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
   | `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` | Where outputs go                                                                                                 |
   | `R2_PUBLIC_BASE_URL`                                                     | Public URL of the bucket, returned to the server                                                                 |
   | `ALLOWED_IMAGE_HOSTS` (trellis2)                                         | Comma-separated hosts `image_url` may point to, e.g. your R2 domain. Without it, only `image_base64` is accepted |
   | `TRELLIS2_LOW_VRAM=0` (trellis2, optional)                               | Keep all models on the GPU; faster on 48 GB+ cards                                                               |
   | `FLUX_CPU_OFFLOAD=1` (flux-schnell, optional)                            | Run on 24 GB cards, several times slower                                                                         |

4. Give the server `RUNPOD_API_KEY`, `RUNPOD_TRELLIS2_ENDPOINT_ID` and
   `RUNPOD_REFERENCE_ENDPOINT_ID` (see `.env.example`). `AI_WORKERS_URL` wins if both are set.

Keep endpoints at **0 minimum workers** (scale to zero). An always-on RTX 4090 pod ($0.74/h)
only beats serverless ($1.10/h, billed while busy) once the GPU would be busy about two-thirds
of the time.

### RTX 5090 and other Blackwell cards

The images target CUDA 12.4 (compute capability 8.0, 8.6, 8.9 and 9.0). Blackwell (sm_120) needs
CUDA 12.8+: switch the base image to `nvidia/cuda:12.8.x-cudnn-devel-ubuntu22.04`, install a
cu128 build of PyTorch, add `12.0` to `TORCH_CUDA_ARCH_LIST`, and use a flash-attn build for
that torch version. CuMesh, FlexGEMM and o-voxel must be compiled against that exact PyTorch,
which is what the Dockerfile and `modal_app.py` do; a mismatch shows up as an ABI error at import.

## Cost (estimates to check against real `timings`)

Modal on 29 Sep 2026: L40S $0.000542/s ($1.95/h), plus $0.047 per CPU core-hour and $0.008 per
GiB-hour of memory, so about $2.30/h per worker container as configured.

| Step (Modal, L40S)                                | Time (estimate) | Cost        |
| ------------------------------------------------- | --------------- | ----------- |
| 4 reference images                                | 10–20 s         | ~$0.01      |
| Preview                                           | 20–40 s         | ~$0.01–0.03 |
| Final                                             | 1–2 min         | ~$0.04–0.08 |
| Cold start and 60 s idle, per container scaled up | ~2 min          | ~$0.08      |

The GPU work in a prompt-to-final run comes to about $0.06–0.12. At low traffic each run also
pays for its cold starts: one FLUX and one TRELLIS.2 container, plus a second TRELLIS.2 start if
the final comes more than 60 s after the preview. That makes roughly $0.22–0.36 a run, so the
free $30 covers about 80–140 runs a month while testing, and more once steady traffic keeps
containers warm. RunPod list prices on 29 Sep 2026: RTX 4090 $1.10/h serverless ($0.74/h
always-on), RTX 5090 $1.58/h ($0.99/h), L40S $1.75/h, RTX A6000/A40 $1.22/h; once the free
credits are used up, its cheaper GPUs make each job cheaper.

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
pip install -r workers/requirements-dev.txt
python -m pytest workers/tests                 # the job API and modal_app.py
python -m pytest workers/trellis2/tests        # CPU only; set GLTFPACK_BIN to include gltfpack
python -m pytest workers/flux-schnell/tests
```

Run the three folders separately: they share test file names.

The tests cover input validation, job handling, the checkpoint check, the job API and the
nvdiffrast stand-in (against a brute-force rasterizer and analytic results). Before the first
production deploy, run `workers/trellis2/scripts/compare_nvdiffrast.py` once on a GPU machine
that has nvdiffrast installed (evaluation use) to confirm the stand-in matches it on real
hardware.
