# Forge3D — AI-Generated 4D Experiences: Plan

**Goal:** A user types a prompt (or drops in an image or a short video) and gets a
*realistic, explorable 3D scene that moves over time*: they can walk the camera
anywhere, scrub a timeline, and share it in the feed as a published experience.

"4D" here means **3D + time**: dynamic geometry, motion, and lighting that hold
together from any viewpoint at any moment. This is the gap that today's AI
video (fixed camera path, 2D pixels) and AI image-to-3D (static objects) leave
open, and it is a natural extension of what Forge3D already has: a 3D editor, a
publishable experience format, and an AI generation orchestrator.

---

## 1. Why this is possible now

Four model families have matured and can be chained:

| Capability | What it gives us | Representative work (benchmark before choosing) |
|---|---|---|
| **Video diffusion** | Realistic motion, physics "priors", lighting, people and animals | Veo, Sora, Kling, Runway; open weights: Wan, HunyuanVideo, LTX-Video, CogVideoX |
| **Camera-controlled / multi-view video** | Same moment seen from many angles, i.e. the raw material for 3D | CameraCtrl, ReCamMaster, GEN3C, SV4D, CAT4D-style multi-view video |
| **Feed-forward 3D/4D reconstruction** | Turns frames into geometry in seconds, not hours | DUSt3R/MASt3R, VGGT, MonST3R (dynamic), 4D Gaussian Splatting variants, L4GM |
| **Asset generation, rigging, motion** | Editable meshes that animate and react | Tripo, Meshy (already integrated), Hunyuan3D, TRELLIS; auto-rigging; text-to-motion diffusion |

And on the rendering side, **Gaussian splats now render in the browser** at
interactive frame rates (e.g. Spark, GaussianSplats3D for three.js), which is
what lets photoreal output live inside Forge3D's existing viewer.

Also worth tracking but not building on yet: **interactive world models**
(Genie-style, World Labs, Odyssey) that generate frames in real time from user
input. They may eventually replace parts of this pipeline (see §7).

---

## 2. The core idea: two ways to make 4D, then combine them

There is no single model that outputs a high-quality, editable, web-playable 4D
scene. Instead there are two complementary paths, and the product should use both.

### Path A — "Lift video into 4D" (realism first)

```
prompt / image
   → video diffusion (hero clip, sets the look & motion)
   → multi-view video generation (same clip, N camera orbits, time-synced)
   → 4D reconstruction (dynamic Gaussian splats: per-frame or deformation field)
   → clean-up (floaters, temporal flicker, scale/ground alignment)
   → compressed splat sequence streamed to the viewer
```

- **Strength:** Photoreal. Inherits everything video models are good at: skin,
  hair, water, smoke, foliage, cinematic light.
- **Weakness:** Hard to edit; limited to the time window that was generated
  (seconds, not minutes); viewpoints far outside the generated cameras degrade;
  heavy files.

### Path B — "Compose a 4D scene from parts" (control first)

```
prompt
   → LLM scene planner (layout, objects, characters, story beats, camera)
   → asset generation (Tripo / Meshy / open models) per object
   → auto-rig characters + text-to-motion / video-to-motion for animation
   → physics + procedural motion (cloth, particles, wind, water)
   → timeline in Forge3D's SceneData, editable in the editor
```

- **Strength:** Fully editable, interactive, infinitely long, cheap to play,
  fits the existing ECS / editor / collaboration stack.
- **Weakness:** Looks "game-like" rather than "filmed" unless lighting, materials,
  and motion quality are very good.

### Path C — Hybrid (the product we ship)

**Splat environment + mesh actors, composited in one scene.**

- The *world* (a forest, a street, an apartment) comes from Path A or from a
  static image/world-to-splat model: photoreal, mostly static, with localized
  4D motion (water, leaves, crowds in the distance).
- The *subjects* the user cares about editing (characters, vehicles, props,
  interactive objects) come from Path B: meshes with rigs, animations, and
  interaction hooks.
- Lighting is estimated from the splat environment and applied to the meshes so
  they sit in the scene (environment map + shadow catcher).
- Hero subjects can be upgraded to "4D splat actors" when realism matters more
  than editability (e.g. a generated person performing a short action).

This gives users a realistic result immediately *and* something they can
actually work with in the editor, which is Forge3D's differentiator over a pure
video generator.

---

## 3. What makes it look *realistic* (quality bar)

Realism in 4D fails in specific, measurable ways. Each one gets an owner and a metric.

| Failure mode | Fix | Metric |
|---|---|---|
| Looks right from the original camera, falls apart when you orbit | Generate more, wider-baseline views; constrain viewer camera to a "safe volume" with soft limits; inpaint novel views and re-fit | Novel-view consistency score (re-render vs held-out generated view, LPIPS/DINO similarity) |
| Flicker / popping over time | Deformation-field 4DGS instead of independent per-frame splats; temporal smoothing pass | Temporal warp error between frames |
| Floaters and fog | Depth priors, opacity regularization, automatic floater pruning | % of splats outside estimated surface; manual QA |
| Wrong scale, floating objects | Metric depth estimation + ground-plane detection; snap to y=0 | Contact error (objects vs ground) |
| Meshes look pasted in | Light estimation from environment, contact shadows, color matching, matching camera grain/DOF | A/B human preference: "which is real?" |
| Robotic character motion | Motion from video-to-motion (pose extraction) or motion diffusion, not hand-keyed loops; foot-contact cleanup | Foot-skate metric; human rating |
| Physically implausible motion | Physics sim for secondary motion; use video model only for primary action | Human rating |

**Evaluation harness (build in Phase 0):** a fixed set of ~100 prompts across
categories (nature, urban, characters, products, fantasy), auto-scored on the
metrics above plus a weekly blind human preference panel. Every model swap is
judged on this harness, not on demos.

---

## 4. How it fits the current codebase

What exists today and what changes:

| Area | Today | Change |
|---|---|---|
| Scene format (`packages/shared/src/types/scene.ts`) | Static entities: transform, mesh, camera, light | Add time: `animation` component (clips, tracks, keyframes), `splat` component (static or 4D splat asset ref), `rig` component, `physics` component. Bump `SceneData.version` with a migration. |
| Experience formats (`types/experience.ts`) | `turntable \| animated \| video \| interactive`; camera keyframes only | Add `'4d'` format: duration, loop mode, timeline markers, camera "safe volume", optional guided camera path, interaction defs that can trigger clips. |
| AI orchestrator (`apps/server/src/services/ai/`) | Single job: `text-to-3d \| image-to-3d` via Tripo/Meshy, polling | Generalize to a **pipeline DAG**: jobs with typed inputs/outputs (`video`, `multiview`, `splat4d`, `mesh`, `rig`, `motion`), retries per stage, partial results, cost accounting per stage. Keep the provider interface; add capability types. |
| `ai_generations` table | One row per provider call | Add `pipeline_runs` + `pipeline_stages` tables (parent run, stage type, provider, status, cost, artifacts) so users see progress per stage and we can resume. |
| Renderer (`packages/engine/src/renderer/`) | Three.js WebGPU/WebGL meshes | Add splat rendering (evaluate Spark vs GaussianSplats3D vs custom WebGPU), depth-correct compositing of splats + meshes, time-driven playback of 4D splats, streaming loader. |
| Editor (`TimelinePanel.tsx`) | Camera keyframes only | Becomes a real timeline: per-entity clip tracks, scrub/play, regenerate a segment, "extend 5s" button. |
| Viewer (`apps/client/src/viewer/`) | Builds meshes from `ExperienceData` | Load splat + mesh scene, stream 4D chunks by time, quality tiers by device. |
| Storage / limits | `MAX_FILE_SIZE_MB = 100` | 4D assets need chunked, compressed, CDN-streamed storage with separate budgets per tier (see §6). |

---

## 5. Phased roadmap

Each phase ends with something users can see. Durations assume a small team
(2–3 engineers + 1 ML/graphics engineer) and should be re-estimated after Phase 0.

### Phase 0 — Spike & evaluation harness (3–4 weeks)
- Stand up GPU workers (start with a serverless GPU provider; avoid owning
  hardware until utilization is known).
- Run 3–4 candidate pipelines end-to-end offline on the 100-prompt set:
  1. image → static splat world (baseline)
  2. video model → multi-view video → 4DGS
  3. single video → dynamic reconstruction (MonST3R / 4DGS-from-monocular)
  4. LLM planner → Tripo/Meshy assets → rig → motion (Path B)
- Measure quality (§3), wall-clock time, GPU cost per second of output, output size.
- Check **licenses** of every open model for commercial use; record in a model registry.
- **Exit:** a written go/no-go per pipeline with numbers, and a chosen first stack.

### Phase 1 — Static photoreal worlds + animated mesh actors (6–8 weeks)
The fastest route to "wow" with low risk.
- Splat rendering in the engine and viewer; splat component in `SceneData`.
- Prompt/image → static splat environment generation.
- Existing Tripo/Meshy generation + auto-rigging + a motion library / text-to-motion for characters.
- Lighting match between splat world and meshes.
- Timeline with mesh animation clips; `'4d'` experience format (with static world).
- **Exit:** users can publish "a fox walking through a photoreal misty forest" and orbit it freely.

### Phase 2 — Living environments (6–8 weeks)
- Localized 4D motion in worlds: water, foliage, fire, smoke, clouds, distant crowds
  (via region-masked 4D splats or procedural/particle layers).
- Pipeline DAG + `pipeline_runs` tables; per-stage progress UI; resumable jobs.
- Streaming 4D playback with quality tiers.
- **Exit:** scenes feel alive, not frozen, while staying under the mobile budget.

### Phase 3 — Full 4D from video (8–12 weeks)
- Prompt → hero video → multi-view video → 4D splat scene (Path A end-to-end).
- "Upload your own video → explorable 4D" (user capture is often higher quality than generation).
- Splat actors: generated people/animals as 4D splats, composited into worlds.
- Camera safe-volume authoring and guided paths for viewers.
- **Exit:** a 5–10s generated moment that can be viewed from any angle within the safe volume and passes the realism harness threshold.

### Phase 4 — Directable, longer, interactive (ongoing)
- Segment-level regeneration ("redo seconds 3–5", "make it rain"), temporal extension.
- Story mode: LLM breaks a prompt into shots/beats; each beat is a 4D segment; transitions.
- Interactions trigger clips (tap the door → it opens), building on `InteractionDef`.
- Audio: spatial sound generation synced to the timeline.
- Remix: fork someone's 4D experience, swap the subject, keep the world.
- **Exit:** user-directed multi-shot 4D stories shareable in the feed.

---

## 6. Infrastructure & delivery

- **Compute:** GPU inference is the main cost. Route each stage to the cheapest
  provider that meets quality: hosted APIs for frontier video models, open
  models on our GPU workers for reconstruction and anything we need to fine-tune.
  Queue with priority tiers (free vs paid), cache by prompt+seed, reuse worlds
  across generations.
- **Credits:** price per stage, not per job, and show the estimate before running
  (the `creditsUsed` column already exists; extend to per-stage).
- **Asset format:** compressed splats (e.g. SPZ or quantized PLY) for static
  worlds; for 4D, a canonical splat set + compressed deformation/keyframe stream
  chunked by time so playback can start before download finishes.
- **Device budgets (targets to validate in Phase 0):**
  - Mobile: ~0.5–1M splats, ≤30 MB initial, 30 fps
  - Desktop: ~2–4M splats, progressive streaming, 60 fps
  - Fallback: pre-rendered video of the guided camera path for weak devices and feed previews
- **Feed previews:** auto-render a short video along the guided path for the
  feed card; load the full 4D only on open (ties into the existing thumbnail service).

---

## 7. Build vs. buy, and what to watch

- **Buy/API:** frontier text-to-video, initially. Our moat is not the video
  model; it is turning outputs into editable, shareable, explorable experiences.
- **Build:** the pipeline orchestration, reconstruction tuning, splat/mesh
  compositing, streaming player, editor timeline, and the evaluation harness.
- **Fine-tune later:** multi-view/camera-controlled video on our own successful
  generations (with user consent) once we have volume.
- **Watch quarterly:** native 4D generators and real-time world models. If one
  outputs good, explorable 4D directly, it slots in as a new provider in the
  pipeline DAG. The editor, format, viewer, and feed stay the same. Design
  every stage boundary so it can be swapped.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GPU cost per experience too high for free tier | Static worlds + mesh actors (Phase 1) are cheap; reserve full 4D for paid credits; aggressive caching and world reuse |
| Quality cliff when camera leaves generated views | Camera safe volume, soft boundaries, guided paths; novel-view inpainting |
| Huge files / poor mobile playback | Tiered budgets, streaming chunks, video fallback |
| Model licenses forbid commercial use | Model registry with license status reviewed before any production use |
| Deepfakes / real-person likeness / IP | Prompt and output moderation (extend `moderation` router), block real-person likeness generation from photos without consent, provenance metadata (C2PA) on exports |
| Provider lock-in or a provider shuts down | Provider interface per stage; always keep ≥2 providers or an open-weights fallback per stage |
| Fast-moving research makes choices obsolete | Evaluation harness + swappable stages; re-benchmark quarterly |

---

## 9. Success metrics

- **Quality:** ≥60% of blind raters prefer our 4D output to a same-prompt video
  baseline when allowed to move the camera; realism harness scores trending up per release.
- **Speed:** Phase 1 world + actors in < 3 min; full 4D (Phase 3) in < 15 min.
- **Cost:** GPU cost per published experience within the credit price with healthy margin.
- **Engagement:** 4D experiences vs static ones on view time, shares, and remix rate.

---

## 10. Immediate next steps (next 2 weeks)

1. Pick GPU worker provider; deploy one open video model and one reconstruction model.
2. Assemble the 100-prompt evaluation set and scoring scripts.
3. Prototype splat rendering inside `packages/engine` alongside existing meshes (depth-correct).
4. Draft the `SceneData` v2 schema (animation, splat, rig components) and the `'4d'` experience format.
5. Draft the pipeline DAG interface as a generalization of `AIProvider` / `AIOrchestrator`.
