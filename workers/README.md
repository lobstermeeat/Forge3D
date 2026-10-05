# AI workers

Orainge generates 3D models with its own models on serverless GPUs, billed only while they
work. There are no third-party AI APIs involved.

| Worker                          | Model                         | Job                                           | GPU                                              |
| ------------------------------- | ----------------------------- | --------------------------------------------- | ------------------------------------------------ |
| [`trellis2/`](trellis2)         | TRELLIS.2-4B (MIT)            | image to textured GLB: previews and finals    | 24 GB+ Ampere/Ada/Hopper (L40S, RTX 4090, A100…) |
| [`pixal3d/`](pixal3d)           | Pixal3D (MIT) on TRELLIS.2    | image to textured GLB: finals, off by default | 48 GB (L40S); finals peak at 28–31 GB            |
| [`flux-schnell/`](flux-schnell) | FLUX.1 [schnell] (Apache-2.0) | text to reference images                      | 48 GB (L40S, A6000, A40)                         |
| [`multiview/`](multiview)       | MV-Adapter on SDXL 1.0        | picture to six views, off by default          | 24 GB (A10G)                                     |
| [`painter/`](painter)           | Qwen-Image-Edit-2511 (Apache-2.0) | a final's texture painted from ten views, off by default | 80 GB (H100)                            |

They run on [Modal](https://modal.com) (`modal_app.py`, the simplest way to start) or on
RunPod serverless (the Dockerfiles; the multiview and Pixal3D workers have none yet). Both hosts
speak the same job protocol, so the server talks to either through `apps/server/src/services/ai`
(`SelfHostedProvider`).

## The flow: spend GPU time only on results people keep

1. **Text prompt → 4 reference images** (FLUX.1 [schnell], a few seconds). The prompt is wrapped
   in a product-shot template: one centred object, plain background, soft light, 3/4 view from
   slightly above. That is the input image-to-3D handles best.
2. **User picks one → preview** (`mode: "preview"`): TRELLIS.2 at 512³, 30k triangles, 1K
   textures.
3. **User keeps it → final** (`mode: "final"`, _same seed_): 1024³ cascade, 100k triangles,
   2K textures. The same seed gives the same coarse structure, so the final refines the preview
   the user approved. (On Modal, `ORAINGE_FINAL_MODEL=pixal3d` makes finals with Pixal3D instead:
   The recipe, below. Phase 6's re-test kept TRELLIS.2 as the default.)
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
would paint a different shape, or too little would change, the texture is left as the model made it,
and the result's optional `projection` field says why (`applied`, `reason`, `iou`, `colour`, `pose`,
`seconds`). Previews skip it: it adds a few seconds and the preview is only for choosing. With the
recipe on, Pixal3D's finals go through the same export, after they are levelled (The recipe, step 4).

**Floating pieces can be dropped** (`trellis2/forge3d_worker/cleanup.py`, `Preset.drop_floaters`, off by
default). TRELLIS.2 sometimes leaves a bit of a model in the air beside it: one of Phase 5's dragons has
the tip of its tail hanging loose. With the flag on, the export removes a connected piece right after
`to_glb` (before the projection) only when it is both small, under 8 % of the surface, and apart, more
than 3 % of the bounding-box diagonal from what stays. Pieces within that gap of each other count as one,
so a part reaches the model through the parts it touches (a basket through its ropes), and the floaters
together never take more than 8 % of the surface (more would be several separate things, not floaters).
Distances are between surfaces, so a piece resting on one large flat triangle touches it.

On Phase 5's 39 finals it drops the dragon's tail tip (two back-to-back sheets, 5.7 % of the surface,
4.9 % of the diagonal away) and three specks hidden inside a balloon's envelope and a cabin, and nothing
else. The real parts that come loose sit much closer: a fox's tail tip 0.9 % away, a chair's armrests
0.75 %, a balloon's pilot and burner up to 1.4 %; sprinkles, pearls and chopsticks touch the model
(within 0.35 %). It takes 0.1 to 1.3 s per final on one CPU core.

**A generation's shape can be retextured** (for experiments: `Trellis2Runtime.retexture`). After
`generate()`, `last_latent` holds the shape latent TRELLIS.2 sampled (a few MB, on the CPU), with the
picture, its cutout and the seed. `retexture(sampler_params=…, seed=…, views=…)` samples only the
texture flow again on that shape, so the geometry stays the one approved: `sampler_params` go over
pipeline.json's (12 steps, guidance strength 1, guidance interval [0.6, 0.9]), and `views` (other sides
of the object) steer the texture flow alone, as job views steer every flow. Without a seed it draws the
noise the generation's own texture was drawn from, so `retexture()` alone gives that texture back and
settings can be compared on one noise. The mesh goes through `export()` like any other.

**Texture options** (`mode: "textures"`; see Texture options under Job contracts). On the sides the picture
doesn't show, TRELLIS.2's texture is a lottery. In Phase 7's three sets of rolls (the 20 test prompts each
time), the final's own texture was publishable for 35 of 60, and the best of four textures on the same shape
for about 47. So after a final, the worker makes three more textures for the same shape with `retexture`,
and the creator picks one.

## The painter: finals painted from views round the model (Phase 8), off by default

TRELLIS.2 gets a product's shape right, but its texture comes from a coarse voxel field: soft and blotchy, paler
than the picture, with the picture's shadows and reflections copied in, and the sides the picture doesn't show
made up (a garbled Coca-Cola script, an arcade machine's sides a different red from its front). The painter
paints the final's texture again from views round the model, the way a 3D artist would texture it from product
photos:

1. **Views.** The final's mesh (to_glb's, before the picture's projection) is rendered with its own texture from
   ten cameras: eight round it 15° above the horizon, the first where the picture was taken from, then the top
   and the bottom (`trellis2/forge3d_worker/paint.py`, `ring`).
2. **Painting.** Qwen-Image-Edit-2511 (Apache-2.0, with lightx2v's 8-step Lightning LoRA, Apache-2.0) turns each
   render into a clean product photo of the same object from exactly the same viewpoint
   (`painter/painter_worker/views.py`: the render is its only picture; the prompt names the side and the
   picture's main colours). Given the picture as well, it copied the picture's viewpoint instead (Phase 8, runs
   4 and 5). Each painted view must keep the render's outline (silhouette IoU at least 0.9 after the best small
   shift and scale) and draw nothing the render doesn't have (`novelty`, at most 0.2); a view that fails gets
   up to three seeds, then is left out. A bottom view whose render is dark isn't painted (the model turned two
   cars' dark undersides into a second roof). Each view goes into the texture before the next is rendered, so
   the views agree.
3. **Colour.** A view's brightness and saturation (never its hue, so grey stays grey) are matched to the other
   views where they overlap and to the picture's own colour where the picture saw the surface well, with its
   shading taken out (`joint_tone`: one least-squares solve for all views, robust to details only one view drew).
4. **Bake.** Each texel takes its colour mostly from the view that sees it most squarely (`select_weights`),
   leaving out views whose brightness there disagrees with the others (a highlight one view drew), into a
   4096 × 4096 base colour on the final's own UV layout (the 2048 texture scaled up first: at 2048 the bake lost
   the wheel spokes and grille slats the views drew). Then the export carries on as for any final: the picture's
   projection on top, the shading normals, the glass, gltfpack.

It runs on its own GPU: the `Painter` class in `modal_app.py` (an H100: the model keeps 58 GB of weights
resident). The Trellis2 container sends it the final's mesh, picture and prompt as one kit and waits for the
texture (`trellis2/forge3d_worker/painting.py`); a painter that fails or can't be reached costs the final only
the wait, and the final keeps TRELLIS.2's texture. A final asks for it with `"paint": true` (Job contracts); the
server does so with `AI_PAINT=1` (The Studio's AI panel). Ten views take about 50 s, the bake and colour match a
few more, and up to 40 s more where views need second and third seeds; loading the model from cold takes one to
two minutes, which the server starts while the creator looks at the preview.

**Phase 8's blind test.** Thirteen objects (six of Phase 2's prompts and seven real products, among them the
founder's BMW) were made as today's finals and then painted, on the same shapes. Four reviewers who didn't know
which was which compared the two textures against the picture, from six sides and three close-ups (two of them
saw every pair the other way round). They chose the painted texture 35 times in 52 and it won 8 objects outright
(all four reviewers on the BMW, the camera, the skateboard, the iPhone, the Lamborghini, the Coca-Cola bottle and
the guitar), today's 4 (the arcade machine, whose coral sides the first painter made maroon, since fixed; the
cartoon car and the watch, where the views drew glare and an engraved case back; the PS5 controller, nearly
identical either way).

What it doesn't fix: the shape (TRELLIS.2's), the views' studio reflections on glossy roofs and bonnets (pale
patches), and parts a view invents that the picture doesn't show (the watch's engraved back); objects the editing
model likes to turn to a catalogue angle (sneakers, watches) lose some views, and keep TRELLIS.2's texture there.

## The recipe: Pixal3D finals, off by default

Phase 5's finals, made by TRELLIS.2 alone, invented wrong backs: the arcade machine's back was a dark
smear with a ghost of its front, the camera's back caved in or sprouted parts a camera doesn't have,
and the shield's back carried a crude ghost of its lion. In the Phase 6 experiments, Pixal3D
(TencentARC; TRELLIS.2 with pixel-aligned image features) fixed most of those backs when it was
given the picture as its one view and its model was levelled afterwards. That is the recipe below. Phase 6's
re-test then found it no better overall, so **finals stay TRELLIS.2's by default**. The recipe stays
in the code, off, for the objects it helps and for further work; `ORAINGE_FINAL_MODEL=pixal3d` turns
it on (Turning it on, below).

**Phase 6's re-test.** The recipe made 40 finals: the twenty prompts of `test-sets/phase2.txt` from
the same pictures, on Phase 4's seeds and on the next seeds, in a staging app (nothing deployed). All
40 finished, all made by Pixal3D. The thin rule (step 3) picked the single-view weights for the shield
and the skateboard on both seeds, for the pistol on both (0.15 and 0.12), and for the guitar on the
next seed (0.09).

- Graded independently with the rubric of Phases 4 and 5 (four reviewers): 11 of 20 publishable on
  Phase 4's seeds (Phase 5: 12), 10 on the next seeds (Phase 5: 11), and 13 with a second try
  (Phase 5: 13).
- Blinded side by side, by four more reviewers who each judged both finals of a pair, made from the
  same picture and seed and shown in random order: Phase 5's TRELLIS.2 final was preferred in 24 of 40
  pairs, Pixal3D's in 16. With those reviewers, Phase 5 had 15 of 20 publishable with a second try,
  against Pixal3D's 11.
- Pixal3D won on the made-up backs of the arcade machine and the shield, on the skateboard's print, and
  on polish (the pistol, the knight, the fox). The camera's back stayed wrong.
- It lost on the picture's highlights and shadows baked into the texture (the chair's white glare, the
  ramen bowl's near-black bottom), on duller or blotchy colour (the sneaker, the car, the balloon, the
  books), on loose floating bits (the donut's sprinkles, the dragon's tail tip, debris under the
  cabin) and on glass (the potion, the bubble tea).

With the recipe on, a final made from the picture alone (every final unless the server runs with
`AI_MULTIVIEW=1`; see The Pixal3D worker) goes through these steps
(`pixal3d/pixal3d_worker/pipeline.py`, `Pixal3DRuntime.generate`). They run in the TRELLIS.2 worker's
container, so the server's contract is the same either way: it asks the `trellis2` worker for a
`"final"`. Previews are always TRELLIS.2's `512` pipeline.

1. **TRELLIS.2's preview at the job's seed.** TRELLIS.2's `512` pipeline on the same picture and
   seed (the preview the user kept, made again), exported as previews are.
2. **The picture's camera.** The projection's silhouette search places the picture against that
   preview, which gives the camera's elevation and roll (`pixal3d_worker/level.py`,
   `estimate_pose`); MoGe-2 gives its field of view. The search must pass the projection's own gate
   (silhouette IoU at least 0.93, and no camera far from the winner that fits as well but sees
   another shape, unless it agrees on the tilt within 5°), and a camera placed more than 10° below
   the horizon isn't trusted. A failed gate, or a failed preview, means no levelling, and the result
   says why.
3. **Pixal3D's multi-view weights, with the picture as their one view.** On the side the picture
   doesn't show they invent far less than Pixal3D's single-view weights (a plain back on the helmet,
   a proper rear on the car, a white bowl all round the ramen). Thin, flat objects are the exception:
   the multi-view weights made the shield a hollow tray and the skateboard a doubled deck, and the
   single-view weights build both cleanly. The preview decides (`pixal3d_worker/thin.py`): when the
   smallest extent of its axis-aligned bounding box is at most `PIXAL3D_THIN_RATIO` (default 0.20;
   `none` turns the rule off) of the largest, the single-view weights build the final. On Phase 4's
   seeds of the twenty-prompt test set (`test-sets/phase2.txt`) that is the shield (0.10), the
   skateboard (0.14) and the pistol (0.15), which looks as good either way; next come the arcade
   machine (0.37) and the sneaker (0.40), and everything else is at least 0.46. The box is
   axis-aligned on purpose, so a flat object pictured on the diagonal doesn't count as thin: the
   guitar measured 0.89 on Phase 4's seed, and the multi-view weights built it cleanly in the
   experiments. Its preview on the next seed lay flatter (0.09), so there it got the single-view
   weights. The single-view flow models wait in RAM and are swapped onto the GPU for a thin object
   (a few seconds each way); the decoders, DINOv3, NAF and the background remover are shared.
4. **Levelled.** Pixal3D builds the model aligned to the picture's camera. It has no notion of
   gravity, so an object pictured from 30° above would come out leaning 30° towards the viewer. The
   mesh is turned back by the camera's elevation and roll before the projection
   (`pixal3d_worker/level.py`; tilts under 1° are left alone).
5. **The usual export**, at the final's settings: `to_glb`, unpremultiply (texels the texture pass
   returned darkened by a stray alpha), the picture projected onto the side it shows, smoothed
   shading normals, gltfpack.

**With the recipe on, the final no longer strictly keeps the approved shape.** The preview is
TRELLIS.2's and the final is Pixal3D's rebuild from the same picture and seed: the same object, but it
can differ in detail from the preview the user kept.

**Turning it on, and fallbacks.** Fetch Pixal3D's weights first (Deploying on Modal, step 4), then
deploy with the variable set:

```sh
ORAINGE_FINAL_MODEL=pixal3d modal deploy workers/modal_app.py
```

The variable is read when the app is deployed (or run, for `modal run`), so deploying again without it
turns the recipe off. With the recipe on, TRELLIS.2 still makes a final when Pixal3D's weights are
missing, when Pixal3D failed to load, and when a Pixal3D final runs out of GPU memory even after one
retry in Pixal3D's low-VRAM mode (Pixal3D's models then move off the GPU while TRELLIS.2 makes the
final, and back afterwards). The result's `fallback` says why each time. Other Pixal3D failures come
back as errors, as TRELLIS.2's do. Results name the model that made them in `model`. Texture options are
TRELLIS.2's alone, so with the recipe on they are refused (Texture options, under Job contracts).

**One container, two models.** With the recipe on, the TRELLIS.2 container holds both (`ModelPool` in
`modal_app.py`); with it off, TRELLIS.2 alone. TRELLIS.2 loads first and stays on the GPU, so previews
start as soon as it is up. Pixal3D is built in a background thread (about 90 s) and the first final
waits for it. When Pixal3D takes its first final, TRELLIS.2 goes to sleep for good: its models move to
RAM and it runs in upstream's low-VRAM mode (each model on the GPU only for its stage). Later previews
take a few seconds longer, and the recipe's own preview runs that way anyway; nothing is swapped back
and forth between jobs. A TRELLIS.2 final made while it is asleep goes straight to the `512` pipeline
if it runs out of memory (see `pipeline` under Job contracts). Pixal3D's finals peak at 28–31 GB of
GPU memory on an L40S and take about 45–103 s of GPU in all, the recipe's own preview included. A
final that has to start a container also waits for both models to load. The container has 48 GB of
RAM (`memory=49152`; with the recipe on, TRELLIS.2 asleep and the single-view flow models wait there)
and a 15-minute timeout either way.

## Job contracts

`trellis2` input:

```json
{ "image_url": "https://…", "mode": "preview", "seed": 1234, "request_id": "gen_42" }
```

`image_base64` can replace `image_url`, which is only fetched from hosts listed in
`ALLOWED_IMAGE_HOSTS`. `seed` is optional (a random one is returned). Outputs are stored at
`ai/<request_id>/<mode>-<seed>.glb`, so every result has its own URL and caches never serve a
stale one. With the recipe on, the same input makes Pixal3D's finals (The recipe, above). `mode` is
`"preview"`, `"final"` (the default) or `"textures"`, which makes more textures for a final's shape
(Texture options, below).

`views` (optional) adds up to 8 other pictures of the object, such as the multiview worker's, so
TRELLIS.2 doesn't have to invent its back and sides:
`"views": [{ "image_url" | "image_base64", "azimuth": 180, "elevation": 0, "weight"?: 1 }]`, with
azimuth 0 the side the main picture shows. Each is cut out and cropped like the main picture (the same
size limits apply), and all of them steer TRELLIS.2's three flows together (see
`trellis2/forge3d_worker/multiview.py`); the main picture still counts most (`settings.MULTIVIEW`), and it
alone is painted onto the final. Send the same views with the preview and the final, so the final
keeps the previewed shape (with the recipe on, a final's views go to Pixal3D instead; see The Pixal3D
worker, below).
TRELLIS.2 builds what the views show, good or bad, and the Phase 6 tests found that the multiview
worker's four views at 0, 90, 180 and 270 are the ones to send: its 45 and 315 drawings made every
object worse. Generation takes about one more single-picture pass per view (4 views: 3.3 to 4.7 times a
single picture's time on an L40S). Output:

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
  "model": "trellis2",
  "views_used": 0,
  "timings": { "generate_s": 9.8, "export_s": 4.1, "compress_s": 2.2, "upload_s": 0.3 },
  "credits": ["Built with DINOv3", "3D generation: TRELLIS.2 (Microsoft, MIT)"]
}
```

`pipeline` is the pipeline that made the model: `"512"` for previews, `"1024_cascade"` for TRELLIS.2's
finals, `"pixal3d-1024_cascade"` for the recipe's. A TRELLIS.2 job that runs out of GPU memory is
retried once in low-VRAM mode (the models visit the GPU one at a time). A final that runs out even then
is made once more, still in low-VRAM mode, with the preview's `"512"` pipeline and exported with the
final's settings, and reports `"pipeline": "512"`. A final that runs out while TRELLIS.2's models are
already off the GPU (deployed in low-VRAM mode, or asleep beside Pixal3D) goes to that pipeline at once.
The same seed gives that pipeline the shape the user approved in the preview, and its memory use is known
to fit, while TRELLIS.2's cascade has no cheaper setting for a 1024³ final (see `FALLBACK_PIPELINE` in
`trellis2/forge3d_worker/settings.py`).

`model` (Modal only) is the model that made it: `"trellis2"`, or `"pixal3d"` for a final when the
recipe is on. With the recipe on, a final that TRELLIS.2 made instead says why in `fallback`: Pixal3D's
weights are missing, Pixal3D failed to load (its error), or it ran out of GPU memory (its error).

`views_used` is how many of the job's `views` helped make the model: 0 without views, and fewer than
sent when a view is left out: one with no object in it, or one whose object runs off the frame
(TRELLIS.2 crops to the object, so a view clipped at its edge reads as a whole object with a side cut
off, and the model comes out crumpled; `CLIPPED_EDGE` in `pipeline.py`).

Finals also carry `projection`, whether the picture was painted onto the model (see above); its
time is part of `export_s`.

A final can ask for the painter (The painter, above) with `"paint": true` and `"subject"`, what the object is in
the creator's words (the prompt; "object" when left out, as for a photo). Previews and texture options ignore it.
Its result then carries `paint`, and its textures are packed up to 4096 (`painting.SIZE`):

```json
{ "paint": { "applied": true, "size": 4096, "output": "robust", "views": 9, "of": 10, "joint": "picture", "painter_s": 61.2, "seconds": 64.0 } }
```

`views` of `of` went into the texture; `joint` is what their colours were held to (`"picture"`, or `"none"` when
the picture's projection didn't fit). A final that couldn't be painted is made as always and says why:
`{ "applied": false, "reason": "TimeoutError: …" }` (or `"this worker has no painter"`, on RunPod and with the
recipe on). The painting is part of `export_s`. A preset with `drop_floaters` adds `floaters`: how many pieces the model had,
how many were dropped, their faces and share of the surface, and the largest of them.

A final made by the recipe has the same fields, with Pixal3D in its `credits`, and says what the recipe
did. For example:

```json
{
  "mode": "final",
  "triangles": 100000,
  "pipeline": "pixal3d-1024_cascade",
  "model": "pixal3d",
  "weights": "multiview",
  "thin": { "extents": [0.46, 0.81, 1.0], "ratio": 0.46, "threshold": 0.2, "thin": false, "source": "preview", "decided": true },
  "camera": { "fov_deg": 29.4, "tilt": { "elevation": 21.2, "roll": -0.8, "source": "preview" } },
  "pose": { "applied": true, "reason": "found", "pose": { "azimuth": 4.0, "elevation": 21.2, "roll": -0.8, "…": "…" }, "iou": 0.968, "…": "…" },
  "level": { "applied": true, "elevation": 21.2, "roll": -0.8, "recentred": [0.0, 0.0121, -0.0043] },
  "views_used": 0,
  "credits": ["Built with DINOv3", "3D generation: Pixal3D (Tencent, MIT) on TRELLIS.2 (Microsoft, MIT)"],
  "…": "…"
}
```

- `weights`: which of Pixal3D's weight sets built it, `"multiview"`, or `"single"` for a thin, flat
  object.
- `thin`: what the preview measured: its bounding box's `extents` (smallest first), their `ratio`
  (smallest over largest), the `threshold`, whether that made it `thin`, the `source` (`"preview"`), and
  whether the measurement `decided` the weights (not when the rule is off).
- `camera`: the picture's camera: `fov_deg` from MoGe-2, and the `tilt` the model was levelled by
  (`elevation` and `roll` in degrees, 0 when the search didn't pass its gate, and their `source`).
- `pose`: the search that placed the picture against the preview: whether it passed the gate
  (`applied`) and why not (`reason`), the camera it found (`pose`: `azimuth`, `elevation`, `roll`,
  `fov`, `scale`, `shift`), its silhouette `iou` and `colour` score, the best distinct `runner_up`,
  the `rivals` it weighed, the `seconds` the preview and the search took, and the `preview` itself
  (`pipeline`, `bytes`). A failed preview leaves only `applied`, `reason`, `pose` and `seconds`, and
  then there is no `thin`.
- `level`: what the export turned: whether it did (`applied`), by how much (`elevation`, `roll`),
  and, when it did, how far it moved the model back to the origin (`recentred`).
- `views_used` is 0: the recipe builds from the picture alone.

With `AI_MULTIVIEW=1` on the server (see The Studio's AI panel), both the preview and the final also
get the picture's other sides from the `multiview` worker, the same views for both:

```json
{
  "image_base64": "…",
  "views": [{ "image_base64": "…", "azimuth": 90, "elevation": 0 }, …],
  "mode": "preview",
  "request_id": "gen_42"
}
```

Each view has `image_base64` or `image_url`, like the picture; the server sends its own copies inline
(a few MB more per job; RunPod takes at most 10 MB in a `/run` request). The picture stays the main image,
and the projection still paints from it. The result reports how many views the model was built from as
`"views_used": 6`. A worker from before views ignores them and leaves that out, and the server then logs
a warning. TRELLIS.2 steers its flows with the views for both; with the recipe on, the final goes to
Pixal3D instead, which takes the redrawn front as its main view (see The Pixal3D worker).

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
Without R2 each view comes back inline (`"url": null` and `base64`). The server keeps the views whose
`azimuth` and `elevation` are numbers, copies them into its storage and sends them on to the 3D worker
with their angles; the camera and framing are the 3D worker's to know.
Settings: 30 steps (the same views as MV-Adapter's 50 in 62% of the time), guidance 3, the reference
at 90% of the frame; 46 s per picture on an A10G, 19 GiB of GPU memory at the peak.

### Texture options

Phase 7's rolls made three more textures for each of the twenty finals of `test-sets/phase2.txt`, on the
final's own shape, three times (Phase 4's seeds; the next seeds; Phase 4's seeds with other texture noise).
Four reviewers graded every texture, blinded. The final's own texture was publishable for 10, 11 and 14 of
20; the best of the four for 16 each time (15.8, 15.7 and 15.8 when two reviewers pick and the other two
judge the pick). No automatic pick was trustworthy: image statistics (agreement with MV-Adapter's views of
the shape, the four's medoid, ghost, grey-patch and palette scores) agreed with the reviewers barely better
than chance, and a self-hosted judge, Qwen3-VL-8B looking at turntables of the four, picked a publishable
texture for 27.5 of 40 against 25 for the final's own (30B-A3B: worse; the code is on the
`phase7-judge-experiment` branch). So the creator picks (`test-sets/phase2.phase7-textures.json` has the
data). A `"textures"` job makes those textures for a final the creator already has:

```json
{ "image_url": "https://…", "mode": "textures", "seed": 1234, "request_id": "gen_42", "count": 3 }
```

It takes what the final took: the same picture, the final's `seed` (required here: another seed would make
another shape) and the final's `views`, if it had any. `count` is how many textures to make, 1 to 4 (3 if
left out). The worker makes the final's shape again, as the final job did (the final's preset, seed and
views), but doesn't export its texture: the creator has it. `pipeline` (optional) is the final's: send
`"512"` for a final that fell back to it (its result said `"pipeline": "512"`), and the shape is made with
that pipeline at once, as the fallback made it (the final's preset with the fallback's pipeline, through the
same `generate()`; `FALLBACK_PIPELINE` in `settings.py`). The cascade would make another shape, or run out
of memory again first. `"1024_cascade"` is the same as leaving it out, and anything else is invalid input.
Previews and finals ignore `count` and `pipeline`. Then, for k = 1 to `count`, TRELLIS.2's texture
flow alone samples a new texture for that shape (`Trellis2Runtime.retexture`), from the picture alone, with
its noise drawn from seed `seed + 1000 * k` (the rolls' seeds; `TEXTURE_SEED_STEP` in
`trellis2/forge3d_worker/settings.py`). Each texture is exported and packed as a final is (to_glb,
unpremultiply, the picture's projection, shading normals, gltfpack) and stored beside the final's model, at
`ai/<request_id>/final-<seed>-texture-<k>.glb`. A second job with the same input makes the same textures
again (give or take the GPU's own nondeterminism) and stores them under the same keys. Output:

```json
{
  "request_id": "gen_42",
  "mode": "textures",
  "seed": 1234,
  "textures": [
    {
      "texture_seed": 2234,
      "glb": { "key": "ai/gen_42/final-1234-texture-1.glb", "url": "https://assets…/ai/gen_42/final-1234-texture-1.glb" },
      "bytes": 1550172,
      "raw_bytes": 8915432,
      "triangles": 96205,
      "projection": { "applied": true, "reason": "applied", "iou": 0.9755, "…": "…" },
      "export": { "path": "to_glb", "seconds": "…", "captured": true }
    },
    {
      "texture_seed": 4234,
      "glb": { "key": "ai/gen_42/final-1234-texture-3.glb", "…": "…" },
      "export": { "path": "rebake", "seconds": "…" },
      "…": "…"
    }
  ],
  "texture_errors": [{ "texture_seed": 3234, "error": "ConnectionError: R2 unreachable" }],
  "pipeline": "1024_cascade",
  "model": "trellis2",
  "views_used": 0,
  "timings": { "generate_s": 27.7, "retexture_s": 30.6, "export_s": 70.8, "pack_s": 4.5, "upload_s": 0.9 },
  "credits": ["Built with DINOv3", "3D generation: TRELLIS.2 (Microsoft, MIT)"]
}
```

- Each texture has a final's fields: `glb`, `bytes`, `raw_bytes`, `triangles` and `projection` (and
  `floaters`, with a preset that drops them). `texture_seed` is the seed its noise came from.
- `export` says how the texture's model was made. TRELLIS.2's `to_glb` does the same geometry work for
  every texture of one shape (filling holes, remeshing, unwrapping UVs, finding where each texel lies on the
  surface); only the texture changes. So the first texture runs it in full and keeps that work, the
  texture layout (`"path": "to_glb"`, `"captured": true`), and the others only sample their texture at the
  kept texels and build the material as `to_glb` does (`"path": "rebake"`;
  `trellis2/forge3d_worker/rebake.py`). `seconds` is that step's time. A layout serves one shape in one job:
  when it doesn't fit (other voxels, say) or the rebake fails, `to_glb` runs in full and `fallback` says why.
  Finals and previews always run `to_glb` in full, as before, and keep nothing.
- A texture that fails doesn't lose the others: it is listed in `texture_errors` (present only then), and
  the job completes with the rest, in order. When none is made, the job fails as other jobs do (`{"error":
  "generation failed: none of the 3 textures was made: …"}`), and so it does when the shape can't be made.
  After a GPU fault the container is replaced once the job is done, either way.
- `pipeline` is the shape's: `"1024_cascade"`, or `"512"` when the cascade ran out of GPU memory even in
  low-VRAM mode, or when the job asked for `"512"`. Textures for a `512` shape fit the final only if the
  final's own `pipeline` was `"512"` too, so a job for a final that fell back should say so; the server
  checks the two match either way.
- `timings`: the shape's generation, then each step summed over the textures, failed attempts included.
  `views_used` and `model` (Modal only) are as for a final.
- **Pixal3D finals.** Retexturing is TRELLIS.2's, and a recipe final has Pixal3D's shape, so with
  `ORAINGE_FINAL_MODEL=pixal3d` every textures job fails at once with
  `{"error": "texture options need TRELLIS.2 finals"}`, before anything runs (`handle_with_models` in
  `modal_app.py`). That holds in a container whose Pixal3D couldn't load as well, since the final may come
  from one where it did. The Pixal3D worker's own service refuses them the same way.

**Cost:** one shape generation (Phase 5's finals: a median of 28 s on an L40S), then per texture its
sampling (2-7 s; much longer for very detailed shapes such as the books), its export and about 1.5 s of
packing. The first texture's export runs `to_glb` in full (15-25 s); the later ones rebake on its layout in
6-9 s, the same model as a full export of that mesh would give (`ops/exp_rebake.py` on an L40S: no visible
difference on the same mesh; `to_glb` 9-19 s against a rebake of about 3 s). Measured on the arcade machine
and the skateboard: 87 and 56 s for three textures, against 111 and 71 s without the cached layout. That is
about $0.04-0.06 of the TRELLIS.2 container (Cost, below).

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

### Views of a given mesh (experiments only)

MV-Adapter's image+geometry model ("ig2mv", `mvadapter_ig2mv_sdxl.safetensors`) draws six views of a
mesh it is given, from a picture of the object: the views follow the mesh, so they can be baked onto
its texture. `multiview/multiview_worker/geometry.py` runs it as MV-Adapter's
`scripts/inference_ig2mv_sdxl.py` does, except that the caller renders the mesh (upstream renders
with nvdiffrast, which is never installed here; the TRELLIS.2 worker's torch rasteriser,
`rasterize_depth` and `rasterize_faces` in `trellis2/forge3d_worker/projection.py`, can). On
Modal it is the `GeometryViews` class: MultiView's image on an A10G, one container that scales down
after 30 s idle. Nothing in production calls it. A `modal run` script calls it as
`modal_app.GeometryViews().generate.remote(job)` with

```json
{
  "image_base64": "…",
  "control_pngs": ["…12 base64 PNGs…"],
  "prompt": "a wooden shield with a lion",
  "seed": 0,
  "steps": 30,
  "guidance": 3.0,
  "reference_scale": 1.0,
  "control_scale": 1.0
}
```

where only `image_base64` and `control_pngs` are needed (the rest are the defaults shown, except
`prompt`, which is "high quality" by default; steps 1–100, guidance 0–20, scales 0–3). It returns
`{"views": [six base64 RGB PNGs], "seconds", "timings"}`, or `{"error": "invalid input: …" |
"generation failed: …"}`. The views are 768 x 768, drawn on mid-gray with lighting, in the cameras'
order; the caller knows the mesh's masks.

The conventions, copied from upstream's script and its training config:

- **Cameras**, in this order: front, right, back, left, top, bottom (`cameras.py`: `IG2MV_VIEWS`,
  `IG2MV_AZIMUTHS = (0, 90, 180, 270, 180, 180)`, `IG2MV_ELEVATIONS = (0, 0, 0, 0, 89.99, -89.99)`,
  `camera_to_world`). Upstream: `get_orthogonal_camera(elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
  distance=[1.8] * 6, left=-0.55, right=0.55, bottom=-0.55, top=0.55, azimuth_deg=[x - 90 for x in
  [0, 90, 180, 270, 180, 180]])`. They are the views' cameras above (same world, distance, frame and
  pixel grid, row 0 at the top), plus one looking down with the front at the image's top and +X on
  its left, and one looking up with the front at the bottom and +X on the left. The order matters:
  the model's attention runs along rows across the four level views and along columns across the
  back, top, bottom and the mirrored front.
- **The mesh** goes where upstream's `load_mesh(path, rescale=True)` puts it: scaled so that its
  largest |coordinate| is 0.5, without re-centring, and turned from glTF's +Y up to +Z up, a glTF point
  (x, y, z) becoming (x, -z, y) (`mesh_to_world`, `gltf_to_world` for normals). glTF's front (+Z) then
  faces the front camera. TRELLIS.2's finals are centred and mostly have the pictured side there (the
  Phase 5 arcade machine and shield do), but a picture taken from high above can leave it facing up:
  Phase 5's guitar, pictured lying down, has its strings in the top and back views.
- **The control maps** are in that world, not relative to the camera: per view, channels 0–2 are the
  surface position + 0.5 and channels 3–5 the unit normal / 2 + 0.5, clamped to [0, 1], and both are
  0.5 where no surface is (upstream renders with `normal_background=0.0`; positions are 0 there). That
  is `encode_control`, giving the (6, 6, 768, 768) array `draw_views` takes. MV-Adapter's training data
  is encoded the same way (world positions + 0.5; world normals, `use_camera_space_normal` off; 0.5
  for the background). Upstream interpolates the mesh's vertex normals and normalises them per pixel.
- **Transport**: `pack_control` turns the array into the job's 12 PNGs (8-bit RGB): the six position
  maps, then the six normal maps, each in view order; `unpack_control` reverses it. 8 bits keep
  positions within 0.002 units (1.4 px) and normals within 0.004; a TRELLIS.2 final's maps are
  0.4–1.3 MB of base64.
- **Settings**: those of the views above (BiRefNet's cutout of the picture framed by `preprocess_image`
  on mid-gray, ShiftSNR with shift scale 8, the fp16-fix VAE, guidance 3, the negative prompt), 30 steps
  (the script's default is 50; the training config evaluates at 30), and control and reference scales 1.

Checked on MV-Adapter's own demo mesh (`assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.glb`):
its maps rendered this way outline the six views upstream drew for it (`…_mv.png`) at a silhouette IoU
of 0.96–0.98 in every view, against 0.38–0.55 with the rows flipped, top and bottom swapped or right
and left swapped. The weights come with `download_models --which multiview` (3.6 GB more; a volume
filled before Phase 7 needs that run again). Licenses as for the views above: MV-Adapter's code and
weights are Apache-2.0, SDXL's license is CreativeML Open RAIL++-M.

### Views baked into the texture (Phase 7, an experiment)

`trellis2/forge3d_worker/mvtexture.py` is the geometry side of letting MV-Adapter's image+geometry
model (ig2mv, `scripts/inference_ig2mv_sdxl.py`) draw the sides the picture doesn't show: it renders
the final's own shape as the position and normal maps ig2mv draws six views from, and bakes those
views into the base-colour texture. Nothing in production calls it: an experiment sets
`Trellis2Runtime.before_projection`, which runs between unpremultiplying the texture and the picture's
projection (to_glb, unpremultiply, bake, project, shade, export).

- **Its cameras are not the image-only model's**: six orthographic views at the same scale (768 px
  over [-0.55, 0.55]), of the front (glTF +Z), the side on its right (+X), the back, the left side, the
  top (the front at the top of the image) and the bottom (the back at the top), in that order.
- `frame_for(mesh, azimuth)` places to_glb's mesh as upstream's `load_mesh(rescale=True)` does (+Y up
  becomes +Z, the largest coordinate 0.5, not re-centred), after turning it so that the picture's
  azimuth (the projection's `pose`) faces the front camera.
- `control_maps` and `as_control` give the control image upstream renders, (6, 6, 768, 768): world
  positions + 0.5, then world normals / 2 + 0.5, both 0.5 off the mesh.
- `bake_views` writes six drawn views into the texture: per texel a depth test in each view, a weight
  cos³ (none under a cosine of 0.2), fades within 4 px of silhouettes and depth edges, a blend in
  linear light, one colour match to the old texture for all six views; the old colour where the
  weights add up to little, fading in; the views' grey background left out; the change carried into
  the gutters. Each of these is a keyword argument.
- On the Phase 5 arcade machine, its own texture rendered from the six cameras and baked back changes
  by 0.64/255 on average where it is baked (99% of texels within 12.3; the rest sit at sharp edges and
  UV seams), and by 0.24 (99% within 4) from renders twice the size: what is left is resampling. About
  60% of its texels, mostly the inner faces of its panels, are seen by no camera and keep their colour.

### The Pixal3D worker

`pixal3d/pixal3d_worker/service.py` is the `trellis2` contract with Pixal3D behind it, plus views. With
the recipe on, the TRELLIS.2 container hands it the finals (`handle_with_models` in `modal_app.py`); it
has no endpoint or Dockerfile of its own. Besides `trellis2`'s input it takes, optionally, the views
the `multiview` worker draws around the picture:

```json
{
  "image_url": "https://…", "mode": "final", "seed": 1234, "request_id": "gen_42",
  "views": [
    { "image_url": "https://…/view-0.png", "azimuth": 0, "elevation": 0 },
    { "image_url": "https://…/view-90.png", "azimuth": 90, "elevation": 0 }
  ],
  "camera": { "type": "orthographic", "half_extent": 0.55 }
}
```

Views are square RGBA cutouts from level orthographic cameras that share one scale; azimuth 0 is
the picture's own view redrawn, positive azimuth towards the picture's right, 180 the back (the
`multiview` worker's convention). `camera` is optional and says how wide their frame is in their
own units (MV-Adapter's `0.55` by default); it only ends up in the result, because the worker
rescales the views' world from their silhouettes so the object fills Pixal3D's cube the way its
training objects did (`pixal3d_worker/views.py: fit_camera`).

Without views a picture goes through the recipe above. With views (on Modal, a final while the recipe
is on and the server runs with `AI_MULTIVIEW=1`), Pixal3D's multi-view weights build the model from
all of them, without the recipe's preview, levelling or thin rule, since the views are level already.
The redrawn front (azimuth 0), when there is one, is Pixal3D's main view; the user's picture is still
what the projection paints onto the final. The result has `"pipeline": "pixal3d-mv-1024_cascade"`,
`"weights": "multiview"`, `"views_used"` (how many of the job's views went in) and `"camera"`: the
views' angles (`views`), the frame used (`half_extent`, and the job's `given_half_extent`), the
object's longest extent in the views' units (`extent`), whether the frame was `rescaled`, and which
views cut the object off (`cut_off`, when they do). In the Phase 6 experiments a final with six views
took 40–70 s to generate. Pixal3D has no 512 pipeline: a preview sent to this worker is the final's
generation exported lighter (production never sends it one). A job that runs out of GPU memory gets
one retry in Pixal3D's low-VRAM mode.

The experiments build their runtime with `runtime_from_env`, set by `PIXAL3D_*` variables
(`PIXAL3D_WEIGHTS`, `PIXAL3D_LEVEL`: `preview`, `none`, or `given` for a pose the caller sets,
`PIXAL3D_FOV_DEG`, `PIXAL3D_MAIN`, `PIXAL3D_AZIMUTHS`, `PIXAL3D_LOW_VRAM`, `PIXAL3D_THIN_RATIO`).
With the recipe on, the Modal container builds its own in `modal_app.py` (`build_pixal3d`): the
multi-view weights with the single-view set beside them, everything on the GPU, levelled against the
container's own TRELLIS.2. Of those variables only `PIXAL3D_THIN_RATIO`, from the container's
environment, changes it.

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
worker's container ahead of a job without waiting for it. `/trellis2` takes previews, finals and
texture options alike (with the recipe on, its container makes the finals with Pixal3D and refuses
texture options).

1. On Hugging Face, request access to
   [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (Meta approves it
   manually, so do this first) and accept the
   [FLUX.1 [schnell]](https://huggingface.co/black-forest-labs/FLUX.1-schnell) terms. Create a
   read token. The other weights (Pixal3D, MoGe-2, NAF, MV-Adapter, SDXL) need no request.
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

4. Download the weights into the `orainge-models` volume. This runs in Modal's cloud, on a CPU;
   with `--detach` it keeps going if this computer goes offline, and running it again resumes an
   interrupted download:

   ```sh
   modal run --detach workers/modal_app.py::download_models --which trellis2
   modal run --detach workers/modal_app.py::download_models --which reference
   ```

   Add `--which painter` only if the server will run with `AI_PAINT=1` (the painter, 59 GB: Qwen-Image-Edit-2511
   and its Lightning LoRA, every file checked against its pinned sha256). Add `--which pixal3d` only to turn the
   recipe on (step 6). It fetches both of Pixal3D's weight
   sets (about 44 GB: the multi-view set for most pictures, the single-view set for thin, flat
   objects), MoGe-2 and NAF, and reuses the TRELLIS.2 weights' decoders, DINOv3 and BiRefNet, so it
   comes after `--which trellis2`. Add `--which multiview` only if the server will run with
   `AI_MULTIVIEW=1` (it also fetches the image+geometry adapter the experiments use, 3.6 GB). Without
   `--which`, `download_models` fetches all five sets, about 178 GB (Pixal3D's two flow-model sets are
   44 GB of it, the painter's 59 GB).

5. Update the server's database. From Phase 6 on, the server reads the `views` and `views_error`
   columns of `ai_generations` (`apps/server/src/db/schema.ts`) even with `AI_MULTIVIEW` off, so
   push the schema to its database before the updated server runs:

   ```sh
   DATABASE_URL=postgresql://… pnpm --filter @forge3d/server exec drizzle-kit push
   ```

   (In PowerShell: `$env:DATABASE_URL = "postgresql://…"; pnpm --filter @forge3d/server exec drizzle-kit push`.)

6. Build and deploy. This is the only step that needs this computer online: it takes 20–40
   minutes the first time (the TRELLIS.2 image compiles CUDA extensions). If the connection
   drops, run it again; finished build steps are kept.

   ```sh
   modal deploy workers/modal_app.py
   ```

   Give the server the URL it prints (`https://<workspace>--orainge-ai-api.modal.run`) as
   `AI_WORKERS_URL` and the same token as `AI_WORKERS_TOKEN` (see `.env.example`). The deploy
   also prints which model makes the finals: TRELLIS.2, unless you turn the recipe on (The recipe,
   Turning it on) after fetching Pixal3D's weights in step 4:

   ```sh
   ORAINGE_FINAL_MODEL=pixal3d modal deploy workers/modal_app.py
   ```

   Deploying again without the variable turns the recipe off. Turned on without Pixal3D's
   weights, the finals stay TRELLIS.2's and say so in `fallback`.

7. Make a model. This runs entirely in Modal's cloud, and with `--detach` it keeps going if
   this computer sleeps or goes offline:

   ```sh
   modal run --detach workers/modal_app.py::make --prompt "a brass pocket watch" --final
   ```

   A run first downloads any weights it needs that the `orainge-models` volume doesn't have yet
   (FLUX's for a prompt, Pixal3D's only for a final with the recipe on, also when a run is continued
   with `--final`; `make_set` fetches what its runs need once, up front); later runs start within a couple of
   minutes. Each step is saved in the `orainge-outputs` volume under the run's name: the reference
   images, `preview-<seed>.glb`, `final-<seed>.glb` and `progress.json`. If you are still connected
   at the end, they are also copied to `orainge-outputs/<run>/` here. `progress.json` keeps each
   step's timings, triangles and `pipeline`, and for the final also its `projection`, the `model`
   that made it, any `fallback`, and the recipe's `weights`, `thin`, `camera`, `pose` and `level` (as
   in the result, under Job contracts).

   | To…                                  | Run                                                                                                                |
   | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
   | See what's ready, running or failed  | `python workers/modal_app.py status`                                                                               |
   | Continue an unfinished run           | `modal run --detach workers/modal_app.py::make --run <name>`                                                       |
   | Make the final of a previewed run    | the same, with `--final`                                                                                           |
   | Start from your own image            | `make --image photo.png` instead of `--prompt`                                                                     |
   | Choose which picture becomes 3D      | `make --prompt "…" --pictures-only`, then `make --run <name> --pick 3`                                             |
   | Make a whole test set                | `modal run --detach workers/modal_app.py::make_set --prompts <file>`                                               |
   | Try the recipe beside production     | `ORAINGE_APP_NAME=orainge-staging ORAINGE_FINAL_MODEL=pixal3d modal run --detach workers/modal_app.py::make_set …` |
   | Download a run                       | `modal volume get orainge-outputs <name> .`                                                                        |
   | Fetch FLUX while Meta reviews DINOv3 | `modal run --detach workers/modal_app.py::download_models --which reference`                                       |

   A continued run skips every finished step, so finished work is never paid for twice. The GLBs are
   meshopt/KTX2-compressed: open them in the Orainge editor (File › Import model) or another
   viewer that supports those extensions.

   `ORAINGE_APP_NAME` runs the same code under another app name (production's is `orainge-ai`), for
   a staging copy that never touches production's deployment, such as a re-test as an ephemeral
   `modal run` (Phase 6's re-test ran in one). It shares production's volumes: the weights, the
   outputs and the kernel caches. `ORAINGE_FINAL_MODEL` works with `modal run` as it does with
   `modal deploy`.

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

**Settings** (in `modal_app.py`): the TRELLIS.2 and FLUX workers run on an L40S (48 GB), scale to
zero, stay warm for 60 s after their last job (idle time is billed; a cold start takes about a
minute) and are capped at 2 TRELLIS.2 containers and 1 FLUX container to bound spending. The
TRELLIS.2 container has 48 GB of RAM and a 15-minute timeout whether or not the recipe is on (they
are sized for it: a Pixal3D final, its out-of-memory retry and a TRELLIS.2 fallback fit well
inside). The multiview worker runs on an A10G (24 GB; it peaks at 19 GiB) with 1 container.
`TRELLIS2_GPU = "A10"` costs about half as much per second but is slower and has only 24 GB; set
`TRELLIS2_LOW_VRAM = "1"` with it. The recipe is untested on an A10 (Pixal3D's finals peak at
28–31 GB with its models on the GPU). Compiled GPU kernels and FlexGEMM's kernel tuning are kept in
the `orainge-cache` volume, so only the first containers spend time compiling and benchmarking them.

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
no such route, so there the first job after a quiet spell still waits for its container. With the
recipe on, a final that has to start the TRELLIS.2 container also waits about 90 s for Pixal3D to
load.

The server needs `AI_WORKERS_URL` and `AI_WORKERS_TOKEN` (Modal, above) or the RunPod variables,
and the `ai_generations` table as in `apps/server/src/db/schema.ts`
(`pnpm --filter @forge3d/server exec drizzle-kit push` in development; Deploying on Modal, step 5,
for production). Pictures and models are copied into the server's storage (`UPLOAD_DIR`), so they
outlive the workers' outputs. Each user can have 3 models in progress and 30 an hour until credits
exist (`apps/server/src/services/ai/studio.ts`). Each request to the job API gives up after 30 s
(`REQUEST_TIMEOUT_MS` in `providers/jobEndpoint.ts`; `runsync`, which the host holds open, after 150 s),
so no poll hangs on it; a job's state that didn't come in time is asked again on the next poll.

**The other sides (`AI_MULTIVIEW=1`, off by default).** The 3D models invent the sides a picture
doesn't show. With `AI_MULTIVIEW=1` on the server, the picked picture (or the uploaded photo) first
goes to the `multiview` worker, which draws it from 6 sides; the panel shows "Drawing the other
sides", then builds the preview and the final from the picture and the views (with the recipe on,
the final with Pixal3D's multi-view weights), with the views as small thumbnails under the picture
while the preview and the final are made. It is off by default because the views were often wrong
(the arcade machine's knobby back, the camera's second lens), and the models build what the views
show. The views are copied into the server's storage and kept with the
generation (the `views` and `views_error` columns), so the final and Try again use the same ones, and
picking another picture draws new ones.

The views are optional, so they never fail a generation. If their job can't start, fails, comes back
broken or takes longer than 3 minutes (`VIEWS_TIMEOUT_MS`; it is then cancelled), the preview is made
from the picture alone; the server logs why and keeps it as `views_error`, and the panel says the back
is guessed. While FLUX draws, the server starts the multiview worker as well as TRELLIS.2, and starts
TRELLIS.2 again when the views start for users who took over 2 minutes to pick; the panel starts the
multiview worker while the user chooses a photo. A generation drawing its views counts as one in
progress.

On Modal the server calls the job API's `multiview` worker (`/multiview/run`, `/status`, `/cancel`
and `/warm`), which `modal_app.py` registers next to `trellis2` and `reference`. It needs the
multiview weights (`download_models --which multiview`); without them each views job fails and the
model is made from the picture alone. On RunPod, give the server `RUNPOD_MULTIVIEW_ENDPOINT_ID`;
without it the step is skipped.

**Texture options (`AI_TEXTURE_OPTIONS`, on by default).** On the sides the picture doesn't show,
TRELLIS.2's texture is a lottery. In three runs of the 20 test prompts, the final's own texture was good
enough to publish 35 times in 60, and the best of four textures of the same shape about 47 times (Phase
7's rolls). No automatic pick was trustworthy, so the creator picks:

1. The poll that finds the final done returns it at once, so it is in the scene and usable; under it
   the panel says "Making 3 more textures to choose from…" and polls on every 5 s (after a reload too).
   The next poll starts a `"textures"` job on the `trellis2` worker (Texture options, under Job
   contracts) with what the final was made from: the picture, the final's seed and its views, `count` 3,
   and `"pipeline": "512"` when the final fell back to it.
2. About a minute or two later the panel shows "Texture 1 2 3 4". 1 is the final's own texture. Picking
   another swaps the model in the scene in place: the same object where it stands, so it saves with
   the scene, and Undo puts the last one back. The pressed number is the one in the scene. If the model
   was removed, picking one places it again.

The textures are copied into the server's storage beside the final and kept with the generation (the
`textures_status`, `textures_job_id`, `textures` and `textures_error` columns, and `final_pipeline`), so
a restarted server picks a running job up again. They never fail a generation. If their job can't
start or fails, or its textures fit another shape (the job's `pipeline` isn't the final's, as when a
worker from before `pipeline` makes a fallen-back final's textures on the cascade), the panel says
quietly that no more textures could be made, and the server logs why. When some textures fail, the
panel offers the others. Picking another picture drops them, and stops their job if it still runs
(the job is read and cleared in one statement, so one a poll records meanwhile is stopped too). A job,
or its textures, meant for one final never land on the next: the server records them only while that
final is still the generation's.

The job API's `run` isn't idempotent, so the server starts a textures job once. A start that fails
once its request may have reached the workers (a 5xx, a dropped connection, no answer in 30 s) could
have queued the job anyway, so it ends the texture options instead of queueing another on every poll.
Only a start that never reached them (the connection refused, the host not found) is tried again, by
the polls in the 2 minutes after the final (`TEXTURES_START_MS` in `apps/server/src/services/ai/studio.ts`).
A job still running 15 minutes after it started (`TEXTURES_TIMEOUT_MS`) is stopped and ends them the
same way, so the panel never waits for ever. Each final's options take about 1-1.5 minutes of the
TRELLIS.2 container (about $0.04-0.06).

`AI_TEXTURE_OPTIONS=0` (or `false`, `off` or `no`) on the server turns them off. A final that another
model made gets none (on Modal its result's `model` says, e.g. `"pixal3d"` with the recipe on):
retexturing is TRELLIS.2's. Workers with the recipe on (`ORAINGE_FINAL_MODEL=pixal3d`) refuse every
textures job, even for a final TRELLIS.2 made in Pixal3D's place, so turn them off there too.

**Painted finals (`AI_PAINT=1`, off by default).** With `AI_PAINT=1` on the server, every final asks for the
painter (The painter, above) with the creator's prompt as its subject, and the server starts the painter's GPU
while the creator looks at the preview (at most once every 2 minutes per user), so a final rarely waits for the
model to load. A painted final takes a minute or two longer and gets no texture options: they would be TRELLIS.2's
own textures, unpainted. A final the painter couldn't paint (its result's `paint` says why; the server logs it)
is kept as made, with texture options as usual. Painted finals are about twice the size (3 MB rather than 1.5 MB),
for their 4096 textures. It needs the painter's weights (`download_models --which painter`) and the job API's
`painter` route, which `modal_app.py` registers (warming only); on RunPod there is no painter and finals are
made as before.

To try the panel without GPUs, start the server with `AI_WORKERS_MOCK=1`: stand-in workers draw
labelled pictures (rated, with the second always the best) and return a small house model after
a second or two, then the final's house in 3 other colours as its texture options. A prompt with
the word "fail", or a photo under 64 px, shows the error states, and a photo under 256 px gets no
texture options. With `AI_MULTIVIEW=1` too, they draw 6 views of a box (the front orange, the sides
green and the back blue), except for a photo under 128 px, whose model is then made from the photo
alone. It refuses to run in production.

## Deploying on RunPod

The images hold TRELLIS.2 and FLUX only: there is no recipe on RunPod, and results carry no
`model`.

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
GiB-hour of memory, so about $2.30/h for a FLUX container and $2.50/h for a TRELLIS.2 container
(4 cores and 48 GiB of RAM) as configured.

| Step (Modal, L40S)                                                                 | Time (estimate) | Cost        |
| ---------------------------------------------------------------------------------- | --------------- | ----------- |
| 4 reference images                                                                 | 10–20 s         | ~$0.01      |
| Preview                                                                            | 20–40 s         | ~$0.01–0.03 |
| Final                                                                              | 1–2 min         | ~$0.04–0.08 |
| Texture options: the final's shape again, then 3 textures (the first ~30 s, the others ~15 s) | 1–1.5 min | ~$0.04–0.06 |
| Cold start and 60 s idle, per container scaled up                                  | ~2 min          | ~$0.08      |
| Six views (multiview, A10G at ~$1.30/h all-in)                                     | 46 s            | ~$0.02      |
| Final by the recipe, when `ORAINGE_FINAL_MODEL=pixal3d`                            | 45–103 s        | ~$0.03–0.07 |
| Painting a final, when `AI_PAINT=1` (H100 at $0.001097/s, ~$4.85/h with its 8 cores and 64 GiB) | 60–110 s | ~$0.08–0.15 |
| The painter's cold start and 120 s idle, per container scaled up                   | ~3–4 min        | ~$0.25–0.30 |
| Pixal3D loading, when `ORAINGE_FINAL_MODEL=pixal3d` and a final starts a container | ~90 s           | ~$0.06      |

The views run only with `AI_MULTIVIEW=1`. The GPU work in a prompt-to-final run comes to about
$0.06–0.12. At low traffic each run also pays for its cold starts: one FLUX and one TRELLIS.2
container, plus a second TRELLIS.2 start if the final comes more than 60 s after the preview. That
makes roughly $0.22–0.36 a run, so the free $30 covers about 80–140 runs a month while testing, and
more once steady traffic keeps containers warm. With the recipe on, Pixal3D's part of a final is about
a minute of L40S (its generation and export) plus the recipe's own `512` preview, in place of
TRELLIS.2's cascade, so a final costs about the same; a final that starts a container pays about
90 s more while Pixal3D loads. RunPod list prices on 29 Sep 2026: RTX 4090 $1.10/h serverless ($0.74/h
always-on), RTX 5090 $1.58/h ($0.99/h), L40S $1.75/h, RTX A6000/A40 $1.22/h; once the free
credits are used up, its cheaper GPUs make each job cheaper.

Microsoft publishes H100 timings only (about 3 s at 512³, 17 s at 1024³ and 60 s at 1536³,
before export). Cold starts add model-loading time on the first job after scaling from zero.

## Licenses

See [`trellis2/NOTICE.md`](trellis2/NOTICE.md) and [`pixal3d/NOTICE.md`](pixal3d/NOTICE.md). In
short: every component is MIT, BSD or Apache-2.0 except DINOv3, whose license requires showing
**"Built with DINOv3"** (it is in the editor's File › About dialog, and each result carries
`credits` to show next to generated models), and SDXL under the multiview worker (below).

Pixal3D (TencentARC), which makes finals only with the recipe on, is MIT since 2026-05-21; its code
is pinned at commit `f7cf38429b0bd264f1995f0f8743a88b1c728b94` and its weights at revision
`b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`. With it come MoGe-2 and the utils3d it pins (MIT) and
NAF (Apache-2.0), all pinned and installed in the TRELLIS.2 image either way, and it reuses the
TRELLIS.2 worker's decoders, DINOv3 and BiRefNet. The About dialog lists TRELLIS.2 for image to 3D;
a Pixal3D final's `credits` name Pixal3D.

The painter ([`painter/NOTICE.md`](painter/NOTICE.md)), off by default, runs Qwen-Image-Edit-2511 (Qwen,
Apache-2.0) with lightx2v's Qwen-Image-Edit-2511-Lightning LoRA (Apache-2.0), both at pinned revisions, through
diffusers (Apache-2.0). Nothing of it needs crediting in the app.

Never installed: NATTEN (not a licensing matter; the one call NAF makes is computed in PyTorch,
`pixal3d/pixal3d_worker/neighborhood.py`), nvdiffrast and nvdiffrec (research-only; replaced by
`trellis2/forge3d_worker/uv_raster.py`), RMBG-2.0 (non-commercial; replaced by BiRefNet), FLUX.1
[dev] (non-commercial) and Hunyuan3D 2.1 (not licensed in South Korea, the EU or the UK).

The multiview worker ([`multiview/NOTICE.md`](multiview/NOTICE.md)), off by default, runs MV-Adapter
(Apache-2.0; its pipeline code is vendored in `multiview/mvadapter/` without the nvdiffrast-based mesh
tools) on Stable Diffusion XL 1.0, whose CreativeML Open RAIL++-M license has use-based restrictions
that Orainge's terms of service must pass on to users before the worker serves them. The same applies
to MV-Adapter's image+geometry adapter (Apache-2.0), which only the experiments' `GeometryViews` runs.

## Tests

```sh
pip install -r workers/requirements-dev.txt
python -m pytest workers/tests                 # the job API and modal_app.py: the model pool, the fallbacks, make
python -m pytest workers/trellis2/tests        # CPU only; set GLTFPACK_BIN to include gltfpack
python -m pytest workers/flux-schnell/tests
python -m pytest workers/pixal3d/tests         # CPU only (torch, trimesh): the recipe on a fake pipeline
python -m pytest workers/multiview/tests       # torch for the camera checks; no GPU or weights
```

Run the folders separately: they share test file names. Each suite also runs from inside its folder
(`cd workers/pixal3d && python -m pytest`). With `TRELLIS2_SRC` pointing at a TRELLIS.2 checkout, the
trellis2 tests also run the views through TRELLIS.2's own samplers, not only a reduction of them, and
the cached texture layout through TRELLIS.2's own `to_glb` (skipped without it); without diffusers
installed, one multiview camera check is skipped.

The tests cover input validation, job handling, the out-of-memory retry and fallback, shading
normals (on synthetic terraced, boxy and low-poly meshes), the checkpoint check, the job API, the
nvdiffrast stand-in (against a brute-force rasterizer and analytic results), the picture
projection (synthetic models pictured from known cameras: the camera found, the colours painted,
the fall-backs), the floater cleanup (synthetic donuts, bowls, balloons, cabins and dragons: what floats
apart goes, what touches or is big stays, UVs and texture kept), retexturing (the texture flow alone
on the generation's shape, its noise drawn again bit for bit, views, the low-VRAM retry), texture options
(the final's shape once, the rolls' seeds, the count, the keys, one texture or all of them failing, the
refusal where the finals aren't TRELLIS.2's, the timings), the cached texture layout (TRELLIS.2's own
`to_glb` on CPU stand-ins for CuMesh and FlexGEMM, with `TRELLIS2_SRC`: what the capture sees, a rebake
equal to `to_glb` bit for bit, the fallbacks, finals untouched) and the views' bake (MV-Adapter's ig2mv cameras held to its code, the control maps of
a box, boxes baked from views painted per side: hidden sides, the background, the fades, the colour
match, the gutters). For the recipe they cover, on fakes, the default (TRELLIS.2's finals) and the choice
of model, the container's model pool, the fallbacks to TRELLIS.2 and what they report, the weights a
run fetches, the recipe's preview, levelling and its gate, the thin rule and the weight swap,
Pixal3D's cameras and views, the NATTEN stand-in (against NATTEN's own definition) and the weights
scripts' pins. The multiview tests hold its cameras to MV-Adapter's code and cover the reference
picture's preparation and job handling, and for the image+geometry model the control maps' encoding
and transport, the mesh's placement (against upstream's `load_mesh`), the call into the pipeline and
its jobs, all with stand-ins for diffusers and the weights. Before the first
production deploy, run `workers/trellis2/scripts/compare_nvdiffrast.py` once on a GPU machine
that has nvdiffrast installed (evaluation use) to confirm the stand-in matches it on real
hardware.
