# The 4D plan vs the Orainge plan: evaluation and recommendation

*2026-10-12. Sources: `docs/4d-generation-plan.md` (this branch) and `origin/phase9-model` (the Orainge branch). "README" means `workers/README.md` on `origin/phase9-model`. Bare line numbers point to the 4D plan in its sections and to the README in Orainge's.*

## Summary

The Orainge plan is the better base by a wide margin. It is built, its workers run on Modal, it is costed in dollars and reviewed blind, and its licenses are checked. The 4D plan is a strategy document with no code behind it. It was written against `main`, which is 170 commits behind the Orainge branch and 0 ahead. It builds on Tripo and Meshy, which the Orainge branch deleted on 2026-09-29. It assumes a team of three or four, but the history shows two authors: an AI agent (170 commits) and the founder (2). Neither plan has users: nothing shows a production web app or traffic, and the repo's only real prompt is yours.

Still, the 4D plan is right about what Orainge lacks. Orainge makes one static object on a plain background. Since 2026-09-29 the editor has been "viewing-only to protect creator IP", with no GLB export (commit `feat(editor): remove Export as .glb`). So Orainge's value lies in experiences on the platform, not in assets. Yet its next steps point at Tripo, which sells downloadable assets: a blind comparison, then possibly a fine-tune. Nothing generated can play in the "animated" or "interactive" formats.

**What would be better:**

- Keep Orainge's pipeline and review loop, and make the grading sturdier.
- Run the Phase 9 scoreboard with Pixal3D and the painter as arms. Fine-tune only for a measured shape gap.
- Add movement cheapest first: make clips play, animate generated humanoids, then compose scenes.
- Set a launch date; the usage gates below need traffic.

Splat worlds and video-to-4D become experiments behind set gates.

## What each plan actually is

**The 4D plan** wants a prompt to become a realistic, explorable scene that moves. It has three paths:

- **A:** video → multi-view video → 4D Gaussian splats.
- **B:** an LLM planner → generated meshes → rig → motion.
- **C:** a splat world with mesh actors, the product it proposes.

It also adds a harness, a pipeline DAG, a `'4d'` format and rented video APIs (line 206), over 23–32 weeks for "2–3 engineers + 1 ML/graphics engineer" (lines 135–171). None of it exists, and neither branch has splat code. On `origin/phase9-model` the scene format has a `model` component (URL, source, quality, credits; `scene.ts` lines 11–34) but nothing for time: no clips, rig or splat.

**The Orainge plan** is the 1,183-line README; `plan.md` is a stale alpha checklist. FLUX draws four product-shot pictures and the creator picks one. TRELLIS.2 makes a preview. If it is kept, a final on the same seed goes into the scene by URL (lines 19–34, 928–933). Five workers exist, but by default only FLUX.1 [schnell] and TRELLIS.2 run. Pixal3D, multiview and the painter are off (lines 8–12, 201, 961, 1021).

The 20-prompt set was graded in five rounds, at Phases 2, 4, 5, 6 and 7 (`phase2.txt` lines 9–36): 10/20 publishable at Phase 2, 13/20 at Phase 5 with a second seed (the bar is 16), and 16/20 at Phase 7 with the best of four textures. Phase 8's painter won 37 of 52 choices on 13 objects; its verdicts aren't committed, and it ships off (lines 161–173). Phase 9 added a 100-prompt scoreboard, not yet run, and a data audit that has run: 705,214 of 858,669 TexVerse models are CC BY or CC0, and 181,897 are "clean" (`refs/ops/results`: `runs/20261009-212814-ai-ops-data-637e242/data-summary.json`).

## Scorecard

Eight judges each scored both plans from 0 to 10. Two Orainge scores were lowered after the checks below.

| Lens | 4D | Orainge | Why |
|---|---|---|---|
| Technical feasibility | 3 | 7.5 | Orainge's models run, measured; Path A is research |
| Product and user value | 5 | 6 | Orainge works end to end in staging, no users yet; 4D ships nothing for 9–12 weeks |
| Cost and unit economics | 3 | 8 | Orainge: $0.10–0.18 warm, ~1.5 MB per model; 4D: no dollar figure |
| Codebase fit and execution risk | 3 | 8 | Orainge *is* the codebase; 4D targets a stale branch |
| Quality methodology | 4 | 6 | Orainge drops losers, but grades carry ±2-in-20 noise; 4D's metric is confounded |
| Licensing, legal, safety | 3 | 6 | Orainge vetted licenses, items open; 4D lists non-commercial models; neither screens content |
| Strategic differentiation | 6 | 4 | 4D's moat is the editor and feed; Orainge chases Tripo on a free base model |
| Timeline and sequencing | 5 | 6 | Orainge has a working loop but no roadmap; 4D's order is sensible, its team imaginary |
| **Average** | **4.0** | **6.4** | |

## The 4D plan: right and wrong

**Right**

- **It names the real gap: there is no time dimension.**
  - The viewer keeps only `.scene` and drops the clips (`ViewerPage.tsx:150`, `EmbedPage.tsx:81`).
  - `AnimatedController` looks for clips once, in its constructor (lines 21–28), before the models load (`SceneBuilder.ts` lines 41–46).
  - The interactive format's `playAnimation` fails the same way (`InteractiveController.ts` lines 99–110).
- **It puts the moat in the right place**: "Our moat is not the video model; it is turning outputs into editable, shareable, explorable experiences" (lines 206–207).
- **It orders the risk sensibly**: mesh actors in static worlds come before video-to-4D.

**Wrong**

- **It builds on stale code.** It calls Tripo and Meshy "already integrated" (lines 24, 154). It would generalize `AIOrchestrator` (line 124), which is used only in tests. Its Phase 0 rebuilds GPU workers and a license registry that Orainge already has (lines 139, 147). Only its scene-and-motion harness (lines 99–112) is new.
- **Path A is research**, as its fix, "constrain viewer camera to a safe volume" (line 101), admits.
- **Its license check is one bullet** (line 147). DUSt3R and MASt3R are CC BY-NC-SA, and MonST3R builds on them. Rented video APIs break the no-third-party rule.
- **Its costs and speeds don't hold.** It gives no dollar figure (line 238); we estimate $1–8 of GPU and 30–100 MB per Path A moment. It targets "world + actors in < 3 min" (line 237), yet one Orainge object already takes 2.5–4 minutes of GPU.
- **Its headline metric is confounded.** It compares a scene raters can orbit with a video they can't (line 235).
- **Features with people** (lines 168–169) get one cell of safeguards (line 227).
- **Its exit test, "a fox walking" (line 157), skips the hard part**, which is auto-rigging a generated quadruped. Quadrupeds aren't rare: the fox is Phase 2's first prompt (`phase2.txt` line 42), and 6 of the 17 benchmark characters are animals.

## The Orainge plan: right and wrong

**Right**

- **It switches off what loses**: Pixal3D, "no better overall" (line 201); multiview, "often wrong" (line 961); a judge model that barely beat the default, 27.5 to 25 of 40 (line 487).
- **It spends GPU only on kept results**: $0.06–0.12 warm, $0.22–0.36 cold (lines 1101–1104), before texture options.
- **Its licensing is mostly done** (lines 1133–1136). Three items are still open:
  - SDXL's RAIL++-M terms have to be passed on through terms of service, and there are none yet (lines 1140–1142).
  - DINOv3 is under a custom Meta license (`trellis2/NOTICE.md` line 14).
  - ShareAlike and NoDerivatives data awaits "the attorney review" (`ops/data_audit.py` line 29).
- **Where the models can't be trusted, the creator chooses.**

**Wrong**

- **It makes one object, by design** (lines 21–23). "A small cabin in a snowy forest" was graded publishable 5 times out of 7 with no forest (`phase2*-verdicts.json`). The rubric's "publish" means "Clearly what was asked for, one coherent object (or the group asked for)" (`origin/phase7-judge-experiment:workers/judge/judge_worker/prompt.py`), so the graders failed to apply it.
- **The grades are noisier than reported.**
  - The Phase 2–6 files hold one verdict per run (`verdict`, `note`, `group`), with no grader named.
  - Two groups judged the same 80 models and agreed on 62 (`phase2.phase6-paired.json` against the Phase 5 and 6 files). With either seed, Phase 5 scored 13/20 with one group and 15/20 with the other; Pixal3D scored 13 and 11 (lines 211–217). That ±2-in-20 noise matches the gains between phases (10 → 12 → 13).
  - Counted per prompt, Pixal3D's 24–16 loss is 10–6 with 4 ties.
  - The files point to model graders. The only rubric text is "the one Phase 7's reviewers graded the texture rolls with (phase7/rolls/grader-prompt.md in the scratchpad)". The prompt opens "You are a strict reviewer" (`prompt.py` lines 5–6, 20). No real creator has rated an output.
- **It aims at a commodity, and the direction is yours.** The scoreboard tests "blind against Tripo and the other services, and every new model or fine-tune against the last" (`benchmark.txt` lines 1–2). It is a useful regression test, but 31 of 100 prompts name a brand. The only real prompt is yours, "make a bmw car m3 model blue" (`products.txt` lines 3–4). 8 of the painter's 13 objects were branded, and `ai-ops-shape` (2026-10-09) tuned shapes on the BMW, Huracán, Jordan 1 and Coca-Cola bottle. Without export, matching Tripo means matching an asset creators can't take away.
- **A fine-tune would mostly re-weight familiar data.** The audit lists 44,338 clean products and 25,526 clean characters, but TRELLIS.2 already trained on TexVerse (TRELLIS.2 repo, `data_toolkit/README.md` line 22). The audit skips what's new: TexVerse's `isRigged` and `animation` fields (`data_audit.py` lines 28–37).
- **It understates costs and capacity.**
  - The headline omits texture options, which are on by default (+$0.04–0.06; lines 981, 1093).
  - A painted final pays for two GPUs. The L40S waits 52–112 s through painting and transfer, about $0.04–0.08 (lines 148–159), on top of the H100's $0.06–0.12. With `AI_PAINT=1`, every preview starts the H100, a $0.25–0.30 cold start (`studio.ts` lines 527–529). The gallery prices everything at a flat $2.30/h, with no H100 (`make_gallery.py` lines 42, 110–111), and the costs are still "estimates to check" (line 1082).
  - TRELLIS.2 is capped at 2 containers and FLUX at 1 (`modal_app.py` lines 519, 586). Each container runs one job at a time, and a flow takes 2.3–4.2 minutes of TRELLIS.2 (lines 1091–1093). The platform can serve about 30–50 flows an hour, and two users at `LIMITS.running = 3` (`studio.ts` line 168) can fill it.
  - Nothing writes `creditsUsed`. Pick, keep and retry skip the limits (`studio.ts` lines 261, 287).
  - The texture choice survives only as the scene's model URL, with no event or timestamp (lines 506–508, 991–994).
- **Nothing is planned beyond Phase 9**: no stopping rule, no launch date, and the pre-deploy check (lines 1180–1183) is still pending.

## What would be better

Ground rules: one baseline (`origin/phase9-model`), one pipeline (`StudioWorkers`/`AIStudio`), one review loop (`make_set`, `make_gallery`, verdict JSON), and steps sized for you plus an agent, with something visible every 2–3 weeks.

### Step 0, week 1: measure and clean up

- **Fast-forward `main` to `phase9-model`.** Retire `plan.md`, and replace §4–§7 of the 4D plan with this roadmap.
- **Run the scoreboard.** First, half a day of plumbing: painted and texture-option runs in `make_set`, a competitor script, and anonymised render sheets.
  - Arms: the default, painted finals, and the Pixal3D recipe. Graded independently, Pixal3D was publishable on 11 of 13 everyday prompts against Phase 5's 8, though the paired test tied 13–13 (`phase2.txt` lines 21–25).
  - Competitors: their text-to-3D, and their image-to-3D on Orainge's picture. Count brands separately.
  - Budget: $40–55 of Modal and $20–40 per competitor, over 3–5 days.
- **Grade it reliably.** Use at least two independent graders per item, and store every verdict with its grader, as Phase 7's rolls did (unanimous on 186 of 240). Report agreement, treat gaps within ±2 in 20 as ties, and calibrate once with 5–10 real creators.
- **Write down what each result means before running:**

| Result | Action |
|---|---|
| Orainge ties or wins on game props, characters and home objects | Publish the numbers (without brands) and stop polishing single objects |
| The gap is texture, and the painter closes it | Offer the painter on keep, or as a paid option |
| The gap is shape, and the painter doesn't help | One fine-tune: at most $1.5–2k and 2 weeks, pass bar written first |
| Tripo is as good, at ≤1.5× the cost per publishable model | Your call (see the open questions) |

- **Fix the accounting.** Runs cost $0.10–0.18 warm and $0.26–0.42 cold, and painting adds $0.10–0.20 plus a cold start. Use per-container rates in `make_gallery.py`, and check the figures against two weeks of Modal invoices.
- **Archive ig2mv, mvtexture and GeometryViews**, after moving `_fade` and `_into_gutters` out, since `paint.py` line 58 imports them. Keep Pixal3D until the scoreboard reports.

### Step 1, weeks 1–2: make clips play (no AI)

- **Keep `gltf.animations`** in the viewer, the embed and the editor. Have `AnimatedController` pick up clips as models load, with one mixer per model and clip fields in `ModelData`. The same fix makes the interactive format's tap-to-play work, which the 4D plan's Phase 4 builds on (line 176).
- **Record what creators do.** Backfill texture choices from scenes and `ai_generations.textures`. Add an append-only events table (pick, keep, discard, retry, texture, publish).

### Step 2, weeks 3–6: animated humanoids (the 4D plan's mesh actors)

- **Scope.** Standing humanoids only, about 8–10 of the 17 characters. Animals wait, because rigging quadrupeds is the unsolved part. After the exit test, give one quadruped rigger a week on the fox and the corgi. If it fails, animals stay static.
- **Rigging and motion.** One license-checked rigger with a NOTICE.md, one humanoid skeleton, and CC0 clips (idle, walk, wave). Validate the rigger on rigged CC BY/CC0 TexVerse characters once the audit counts them. No text-to-motion: most of it is trained on AMASS/SMPL, which is licensed for non-commercial use only.
- **How it runs.** An "Animate" side job on a kept final, in the TRELLIS.2 container. It adds no cold start, but it shares the two-container cap.
- **Exit test.** A knight idles, walks and waves, and 6 of 10 humanoids are publishable on `test-sets/motion.txt`. If not, go back to static quality.

### Step 3, weeks 6–9: scene kits (Path B)

- **Plan and place.** A self-hosted LLM (the Qwen3-VL-8B worker from `phase7-judge-experiment`) lists the objects, their heights and a shared style. Rules place them with drop-to-ground. The creator keeps the per-object picks.
- **Cost and wait.** A scene costs about $0.6–1.3. At 60% publishable per object, a fully automatic four-object scene would come out clean only 13% of the time (0.6⁴). A scene counts as one job against the limits but needs 9–17 minutes of TRELLIS.2, so raise the container cap and set a wait target before shipping.
- **First test.** Two weeks on trees, in `test-sets/scenes.txt`.

### In parallel: launch, capacity, credits, safety

- **Launch.** Set a date and start with a private beta. The metrics and gate 2 below need traffic.
- **Capacity and credits.**
  - Size `max_containers` for peak load.
  - Meter real GPU cost into `creditsUsed`, and count pick, keep and retry against the limits.
  - Keep the warm-ups, but start the H100 on keep. Its load mostly overlaps the final's generation; measure the extra wait.
  - Price credits at ≥2× warm GPU cost.
- **Before public listing:**
  - a brand policy;
  - automated prompt and output screening and a DMCA inbox, built on the moderation router (`report`, `listReports`, `removeExperience`);
  - AI labels;
  - an NSFW check;
  - terms of service;
  - a photo retention limit.

### When to start splat worlds (Step 4)

Start only when all three hold:

1. The scoreboard has run and been acted on, and credits are metered.
2. After launch, animated or scene experiences beat turntables on views, likes per view and comments (`trending.ts`) across 200+ experiences, *or* 10% of real prompts ask for scenes.
3. A ~500k-splat SPZ file runs at 30 fps on a mid-range phone, within 30 MB.

Then cap the work at 5 agent days and $150. Build on Spark 2.x, after upgrading three.js from `^0.172.0` to ≥0.180, with worlds from a swappable provider.

**Step 5: Path A and person-video-to-4D leave the roadmap.** Revisit only when all three hold: a commercially licensed stack exists, paid credits exist, and GPU cost per published experience is ≤ ~$1.00. Anything with people in it also needs consent and disclosure.

### What happens to the 4D plan

| Keep | Change | Drop |
|---|---|---|
| Time as the gap; the moat in editor, format and feed | Phase 1 → humanoid mesh actors, no splats | Tripo/Meshy, rented video APIs |
| Path B, engagement metrics, speed targets (re-measured) | Its harness → Orainge's loop plus `scenes.txt` and `motion.txt` | Phase 0's rebuild of workers and registry |
| The failure-mode table, as verdict tags | The DAG → side jobs plus a parent scene record | `pipeline_runs`/`pipeline_stages`, Phase 3, video upload, people as splats |

**Add to Orainge:** a stopping rule, a launch date, telemetry and credits, two graders per item, Wilson intervals per prompt, a "recognisably the named product?" question, a brand policy and safety checks.

## Considered and rejected

- **A fine-tune before the scoreboard**: the audited data mostly overlaps TRELLIS.2's training set.
- **Scene or motion prompts in `benchmark.txt`**: they would spoil the Tripo comparison.
- **Archiving Pixal3D now**: it graded better on everyday prompts in one of two tests.
- **Removing the warm-ups**: 45–100 s more waiting to save a bounded cost.
- **Dropping RunPod**: its GPUs are cheaper once Modal's free credits run out (lines 1107–1110).

## Open questions only you can answer

1. **Is "no third-party AI APIs" a principle or a cost decision?** It decides whether Tripo or world APIs can be used.
2. **Were the "four reviewers" models?** The files suggest so. Is a model panel enough for go/no-go?
3. **Who is the user?** Your prompt and the current work point to product replicas, 34% of the scoreboard. `phase2.txt` names 16–24-year-olds making games, who would want the export the editor removed.
4. **When is launch, and what is the budget** for GPU, raters and lawyers?
5. **Will the team grow?** This roadmap assumes it won't.
6. **How much trademark risk will you accept** from branded replicas in a public feed?