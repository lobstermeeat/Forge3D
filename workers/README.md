# AI workers

Orainge generates 3D models with its own models on serverless GPUs, billed only while they
work. There are no third-party AI APIs involved.

| Worker                          | Model                         | Job                      | GPU                                              |
| ------------------------------- | ----------------------------- | ------------------------ | ------------------------------------------------ |
| [`trellis2/`](trellis2)         | TRELLIS.2-4B (MIT)            | image to textured GLB    | 24 GB+ Ampere/Ada/Hopper (L40S, RTX 4090, A100…) |
| [`flux-schnell/`](flux-schnell) | FLUX.1 [schnell] (Apache-2.0) | text to reference images | 48 GB (L40S, A6000, A40)                         |
| [`multiview/`](multiview)       | MV-Adapter on SDXL 1.0        | picture to six views     | 24 GB (A10G)                                     |

They run on [Modal](https://modal.com) (`modal_app.py`, the simplest way to start) or on
RunPod serverless (the Dockerfiles; the multiview worker has none yet). Both hosts speak the same
job protocol, so the server talks to either through `apps/server/src/services/ai`
(`SelfHostedProvider`).

## The flow: spend GPU time only on results people keep

1. **Text prompt → 4 reference images** (FLUX.1 [schnell], a few seconds). The prompt is wrapped
   in a product-shot template: one centred object, plain background, soft light, 3/4 view from
   slightly above. That is the input image-to-3D handles best.
2. **User picks one → preview** (`mode: "preview"`): TRELLIS.2 at 512³, 30k triangles, 1K
   textures.
3. **User keeps it → final** (`mode: "final"`, _same seed_): 1024³ cascade, 100k triangles,
   2K textures. The same seed gives the same coarse structure, so the final refines the preview
   the user approved.
4. Both passes are packed for the browser with gltfpack: meshopt geometry, WebP colour textures
   and KTX2 (UASTC) metallic-roughness at a quarter of the colour's size. The editor's
   `AssetLoader` decodes them; the decoders are served from `/decoders/` by the client's Vite
   config. (ETC1S, Basis' smaller format, turned neighbouring UV charts into grime and specks
   at the mip levels a model shows at normal viewing distance.)
5. Before packing, each pass gets its own shading normals (`trellis2/forge3d_worker/normals.py`;
   positions don't move). TRELLIS.2 often draws round things as wide flat facets: a potion's bulb
   came out as a 32-sided polygon, 11° from facet to facet, which a glossy material mirrors as a
   grid of blocks. A bilateral filter over the face normals smooths differences up to about 10°
   across 32 voxels of the pass's grid and keeps larger ones, so low-poly facets and panel lines
   stay, and edges sharper than 60° stay crisp (their vertices are split). gltfpack keeps the
   normals as 8-bit octahedral vectors (within 1.2°).

Uploaded images can skip step 1. Images with transparency skip background removal.

**Finals get the picture painted on** (`trellis2/forge3d_worker/projection.py`, `Preset.project_picture`).
TRELLIS.2 bakes colour from a coarse voxel field, so painted detail (a lion on a shield, graffiti)
comes out smeared. After `to_glb`, the worker finds the camera the picture was taken from (a search
over views scored by how well the model's silhouette matches the picture's, with the model's own
colours to tell symmetric views apart), takes the picture's lighting out (its exposure and its
shading, fitted against the texture where both show the same paint), then blends it into the
base-colour texture where the surface faces that camera and is visible to it: fading out at grazing
angles, near outlines, depth edges and misfits, and at specular highlights on glossy surfaces. Where
the texture got a paint's colour wrong (a near-black "cola" milk tea, a red fox pictured orange), that
paint takes the picture's colour, fading round the sides instead of ending in a seam. It never fails
a job: when the silhouettes don't match closely (IoU under 0.93), another camera fits as well but
would paint a different shape, or too little would change, the texture is left as TRELLIS.2 made it,
and the result's optional `projection` field says why (`applied`, `reason`, `iou`, `colour`, `pose`,
`seconds`). Previews skip it: it adds a few seconds and the preview is only for choosing.

## Job contracts

`trellis2` input:

```json
{ "image_url": "https://…", "mode": "preview", "seed": 1234, "request_id": "gen_42" }
```

`image_base64` can replace `image_url`, which is only fetched from hosts listed in
`ALLOWED_IMAGE_HOSTS`. `seed` is optional (a random one is returned). Outputs are stored at
`ai/<request_id>/<mode>-<seed>.glb`, so every result has its own URL and caches never serve a
stale one.

`views` (optional) adds up to 8 other pictures of the object, such as the multiview worker's, so
TRELLIS.2 doesn't have to invent its back and sides:
`"views": [{ "image_url" | "image_base64", "azimuth": 180, "elevation": 0, "weight"?: 1 }]`, with
azimuth 0 the side the main picture shows. Each is cut out and cropped like the main picture (the same
size limits apply), and all of them steer TRELLIS.2's three flows together (see
`trellis2/forge3d_worker/multiview.py`); the main picture still counts most (`settings.MULTIVIEW`), and it
alone is painted onto the final. Send the same views with the preview and the final, so the final keeps
the previewed shape. TRELLIS.2 builds what the views show, good or bad, and the Phase 6 tests found that
the multiview worker's four views at 0, 90, 180 and 270 are the ones to send: its 45 and 315 drawings
made every object worse. Generation takes about one more single-picture pass per view (4 views: 3.3 to
4.7 times a single picture's time on an L40S). Output:

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
  "pipeline": "512",
  "views_used": 0,
  "timings": { "generate_s": 9.8, "export_s": 4.1, "compress_s": 2.2, "upload_s": 0.3 },
  "credits": ["Built with DINOv3", "3D generation: TRELLIS.2 (Microsoft, MIT)"]
}
```

`pipeline` is the TRELLIS.2 pipeline that made the model: `"512"` for previews, `"1024_cascade"` for
finals. A job that runs out of GPU memory is retried once in low-VRAM mode (the models visit the GPU
one at a time). A final that runs out even then is made once more, still in low-VRAM mode, with the
preview's `"512"` pipeline and exported with the final's settings, and reports `"pipeline": "512"`. The
same seed gives that pipeline the shape the user approved in the preview, and its memory use is known to
fit, while TRELLIS.2's cascade has no cheaper setting for a 1024³ final (see `FALLBACK_PIPELINE` in
`trellis2/forge3d_worker/pipeline.py`).

`views_used` is how many of the job's `views` helped make the model: 0 without views, and fewer than
sent when a view is left out: one with no object in it, or one whose object runs off the frame
(TRELLIS.2 crops to the object, so a view clipped at its edge reads as a whole object with a side cut
off, and the model comes out crumpled; `CLIPPED_EDGE` in `pipeline.py`).

Finals also carry `projection`, whether the picture was painted onto the model (see above); its
time is part of `export_s`.

`flux-schnell` input: `{ "prompt": "a brass pocket watch", "count": 4, "seed": 5, "request_id": "gen_42" }`.
Output: `{ "images": [{ "key", "url", "seed" }, …], "prompt", "seconds" }`.

`multiview` input: `{ "image_url" | "image_base64", "seed"?, "prompt"?, "request_id"? }`, the picked
picture (as for `trellis2`; a picture with transparency keeps its own cutout). `prompt` is an optional
short description of the object; without it the model gets MV-Adapter's default caption, "high
quality". **Send the user's prompt**: on the Phase 2 test set a caption that names the object kept its
colours and details in the oblique views, and the hidden side is otherwise a lottery of the seed. A
caption that also says what the back looks like ("a retro arcade machine, a plain flat back panel")
was the only setting that reliably drew a plain back; nothing drew a correct back for a camera.
Output:

```json
{
  "request_id": "gen_42",
  "seed": 1234,
  "views": [
    { "azimuth": 0, "elevation": 0, "key": "ai/gen_42/view-1234-0.png", "url": "https://assets…/ai/gen_42/view-1234-0.png" },
    { "azimuth": 45, "elevation": 0, "key": "ai/gen_42/view-1234-45.png", "url": "…" }
  ],
  "camera": { "type": "orthographic", "image_size": 768, "half_extent": 0.55, "pixels_per_unit": 698.182, "up": [0, 0, 1], "front": [0, -1, 0], "…": "…" },
  "seconds": 46.1,
  "timings": { "cutout_s": 0.4, "views_s": 43.2, "view_cutouts_s": 1.9, "upload_s": 0.6 }
}
```

with six views, at azimuths 0, 45, 90, 180, 270 and 315. Each is a 768 x 768 RGBA PNG: MV-Adapter
draws the views on flat mid-gray (127, the background it was trained on), and BiRefNet's mask of each
view becomes its alpha; the RGB is left as drawn, lighting included (and not consistent between views).
Settings: 30 steps (the same views as MV-Adapter's 50 in 62% of the time), guidance 3, the reference
at 90% of the frame; 46 s per picture on an A10G, 19 GiB of GPU memory at the peak.

### The views' cameras

`multiview/multiview_worker/cameras.py` has these in code (`camera_to_world`, `project`); they are
MV-Adapter's own cameras (`get_orthogonal_camera` as its inference script calls it), and the tests hold
them to its code.

- **World**: right-handed, **+Z up**, the object at the origin, its front facing **-Y**.
- **Azimuth 0 is the picture's view, made level and squared up.** The model was trained to draw fixed
  views of an object from a reference taken from anywhere, so it keeps the side the picture mostly
  shows and turns it to face the 0° camera. A head-on picture matches its 0° view; a three-quarter
  picture comes out straight-on (Phase 2's arcade machine was about 20° off, its car about 35°), so
  the picture sits between two views; a profile picture keeps the profile at 0° (Phase 2's dragon,
  whose face is then at 270°); pictures that look down (FLUX's are taken "from slightly above") are
  redrawn at elevation 0. The picture is not one of the six cameras: give the 3D step the views with
  their poses, and find the picture's own camera by silhouette search (as the projection does) rather
  than placing it at azimuth 0.
- **Positive azimuth moves the camera counter-clockwise seen from above**: the 90° view shows the side
  that is on the right of the 0° view, with the front facing image-left; 180° shows the back, 270° the
  left side. The camera at azimuth a is at 1.8 · (sin a, -cos a, 0), looking at the origin, with image
  right along (cos a, sin a, 0) and image up along +Z (no roll). In glTF terms (+Y up, front +Z) a
  point (x, y, z) here is (x, z, -y) there, and the camera is at 1.8 · (sin a, 0, cos a).
- **Every view is orthographic at the same scale**: 768 px span [-0.55, 0.55] world units both ways,
  698.2 px per unit, with the origin at the image centre: pixel x = 384 · (1 + r / 0.55),
  y = 384 · (1 - u / 0.55) for a point's offsets r and u along the camera's right and up.
- **Framing**: the picture's object is centred with its longer side at 90% of the frame (691 px,
  0.99 units), as MV-Adapter's inference script prepares it, and the 0° view comes out at that size
  (measured 685-700 px) whatever the reference's framing: the scale is the model's, not the input's.
  The other views share it, so an object that is deeper than it is wide overflows the frame in the
  views that show its long axis (a stack of books at 45° and 315°; a car in all four side views, wheels
  cut). A view's cutout touching the frame edge means "unknown beyond here", not the object's edge.

Invalid input (including an image where no object stands out from the background) comes back
as `{ "error": "invalid input: …" }`. Other failures come back as `generation failed: …`. After
a GPU fault the worker also replaces its container (RunPod: `refresh_worker`; Modal: the
container stops taking jobs), because CUDA may be unusable in that process.

Without R2 (below) files come back inline as `base64` instead of `url` (8 MB limit), and the
server passes them around as `data:` URLs. That is fine for trying things out.

## Deploying on Modal

Modal bills GPUs by the second and includes $30 of free compute a month on its Starter plan.
`modal_app.py` defines the workers, a volume for the weights and a small job API
(`job_api.py`) with the same routes as a RunPod endpoint (`/run`, `/runsync`, `/status/{id}`,
`/cancel/{id}`), under `/trellis2`, `/reference` and `/multiview`, plus `/warm`, which starts a
worker's container ahead of a job without waiting for it.

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

3. Store two secrets. The worker token is a random password of at least 32 characters for the
   job API (with a shorter one, the API stays off and answers every request with 503 and the
   reason). The server sends it with every request:

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

   The first run downloads the weights (about 60 GB, 10–30 minutes) into the `orainge-models`
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
   | Choose which picture becomes 3D      | `make --prompt "…" --pictures-only`, then `make --run <name> --pick 3`       |
   | Make a whole test set                | `modal run --detach workers/modal_app.py::make_set --prompts <file>`         |
   | Download a run                       | `modal volume get orainge-outputs <name> .`                                  |
   | Fetch FLUX while Meta reviews DINOv3 | `modal run --detach workers/modal_app.py::download_models --which reference` |

   A continued run skips every finished step, so finished work is never paid for twice. The GLBs are
   meshopt/KTX2-compressed: open them in the Orainge editor (File › Import model) or another
   viewer that supports those extensions.

**Without your computer:** the `AI ops (Modal)` GitHub workflow runs the same commands on
GitHub's servers. Add the repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (Modal >
Settings > API Tokens), put the commands in `ops/run.sh` and push it to the `ai-ops` branch.
The log and anything the script writes to `ops-out/` are saved under `runs/<id>/` in
`refs/ops/results` (`git fetch origin refs/ops/results`), a ref rather than a branch so that
Vercel and similar integrations don't try to deploy it, with the tokens scrubbed. In a public
repository it is public, so what the script writes to `ops-out/private/` is saved only
encrypted, for the holder of the private key matching `ops/results-public-key.pem`
(`ops/seal.sh` explains how to make one and open the results).
`bash ops/gallery.sh orainge-outputs/<set> "<title>"` puts a test set's gallery there.

**Review a test set:** a set is a text file with one prompt, or the path of a photo, per line;
`workers/test-sets/starter.txt` covers eight kinds of game assets, `phase2.txt` twenty prompts
written the way creators type them, and `renders.txt` six photos of CC0 sample models (no FLUX
needed). `make_set` runs every line
(previews and finals; `--no-final` for previews only) with the GPU containers kept warm between
runs, names the runs `<set>-<nn>-<words>` and copies them to `orainge-outputs/<set>/`. Run it
again with the same `--name` to retry what failed. To choose each prompt's picture the way a
user does in the Studio, run the set with `--pictures-only` first, then again with
`--picks "3=2,7=4"` (run number = picture number; runs left out use their best-scored picture). Then
`python workers/gallery/make_gallery.py orainge-outputs/<set> -o gallery/` renders every preview and
final from six angles, with the editor's decoders and lighting, and writes a page showing each
run's reference images, triangles, file size, GPU time and cost, with a 3D viewer (orbit,
zoom, wireframe) for each final. It needs `pnpm install` and Playwright's Chromium, which
`ops/gallery.sh` installs on GitHub's servers; serve the page over http(s) to use the viewer.

**Settings** (in `modal_app.py`): both workers run on an L40S (48 GB), scale to zero, stay warm
for 60 s after their last job (idle time is billed; a cold start takes about a minute) and are
capped at 2 TRELLIS.2 containers and 1 FLUX container to bound spending. The multiview worker
runs on an A10G (24 GB; it peaks at 19 GiB) with 1 container. `TRELLIS2_GPU = "A10"`
costs about half as much per second but is slower and has only 24 GB; set
`TRELLIS2_LOW_VRAM = "1"` with it. Compiled GPU kernels and FlexGEMM's kernel tuning are kept
in the `orainge-cache` volume, so only the first containers spend time compiling and
benchmarking them.

**R2 storage** (for production): put the storage variables from the RunPod table below in a
Modal secret, then deploy with its name in `ORAINGE_R2_SECRET`:

```sh
modal secret create orainge-r2 R2_ACCOUNT_ID=… R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… R2_BUCKET=… R2_PUBLIC_BASE_URL=https://assets.example.com ALLOWED_IMAGE_HOSTS=assets.example.com
ORAINGE_R2_SECRET=orainge-r2 modal deploy workers/modal_app.py
```

In PowerShell the second line is `$env:ORAINGE_R2_SECRET = "orainge-r2"; modal deploy workers/modal_app.py`.
`modal deploy` prints which storage the workers use, so a deploy without R2 doesn't go unnoticed.

## The Studio's AI panel

Library › AI (or Home › Insert › AI model) in Orainge Studio runs the flow above: describe an
object or upload a photo, pick one of the four pictures, get a preview placed in the scene, and
keep it to replace the preview with the final where it stands. Scenes store models by URL (the
`model` component), so they survive saving, reloading, collaboration and publishing.

When the FLUX worker rates its pictures (`score` and `issues` on each image), the panel marks the
highest-scoring one "Suggested" and chooses it to begin with, and a picture's issues show when
the pointer rests on it. The user can still pick any of them.

A container that has scaled to zero takes about 45 s (FLUX) to 100 s (TRELLIS.2) to start, so
on Modal the server starts them early: FLUX when the panel opens, and TRELLIS.2 while the
pictures are drawn. Each user starts each worker this way at most once every 2 minutes. A
container started for nothing costs what a cold start does (about $0.08, see Cost). RunPod has
no such route, so there the first job after a quiet spell still waits for its container.

The server needs `AI_WORKERS_URL` and `AI_WORKERS_TOKEN` (Modal, above) or the RunPod variables,
and the `ai_generations` table as in `apps/server/src/db/schema.ts`
(`pnpm --filter @forge3d/server exec drizzle-kit push` in development). Pictures and models are
copied into the server's storage (`UPLOAD_DIR`), so they outlive the workers' outputs. Each
user can have 3 models in progress and 30 an hour until credits exist
(`apps/server/src/services/ai/studio.ts`).

To try the panel without GPUs, start the server with `AI_WORKERS_MOCK=1`: stand-in workers draw
labelled pictures (rated, with the second always the best) and return a small house model after
a second or two. A prompt with the word "fail", or a photo under 64 px, shows the error states.
It refuses to run in production.

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
   | `TRELLIS2_LOW_VRAM=0` (trellis2, optional)                               | Keep all models on the GPU; faster on 48 GB+ cards. A job that runs out of memory retries once in low-VRAM mode  |
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
| Six views (multiview, A10G at ~$1.30/h all-in)    | 46 s            | ~$0.02      |

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

The multiview worker ([`multiview/NOTICE.md`](multiview/NOTICE.md)) runs MV-Adapter (Apache-2.0; its
pipeline code is vendored in `multiview/mvadapter/` without the nvdiffrast-based mesh tools) on Stable
Diffusion XL 1.0, whose CreativeML Open RAIL++-M license has use-based restrictions that Orainge's terms
of service must pass on to users before the worker serves them.

## Tests

```sh
pip install -r workers/requirements-dev.txt
python -m pytest workers/tests                 # the job API and modal_app.py
python -m pytest workers/trellis2/tests        # CPU only; set GLTFPACK_BIN to include gltfpack
python -m pytest workers/flux-schnell/tests
python -m pytest workers/multiview/tests       # torch for the camera checks; no GPU or weights
```

Run the four folders separately: they share test file names.

The tests cover input validation, job handling, the out-of-memory retry and fallback, shading
normals (on synthetic terraced, boxy and low-poly meshes), the checkpoint check, the job API, the
nvdiffrast stand-in (against a brute-force rasterizer and analytic results) and the picture
projection (synthetic models pictured from known cameras: the camera found, the colours painted,
the fall-backs). Before the first
production deploy, run `workers/trellis2/scripts/compare_nvdiffrast.py` once on a GPU machine
that has nvdiffrast installed (evaluation use) to confirm the stand-in matches it on real
hardware.
