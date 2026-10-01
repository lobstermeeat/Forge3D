#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-pixal3d branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, PIXAL3D line: its own ephemeral app (orainge-exp-pixal3d) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
#
# Run 2. Run 1 (9b1c162) downloaded the weights and made single-view models of 04, 06, 08, 11, 01 and
# 13, synthetic-view models of 04 and 13, and MV-Adapter-view models of 04, 06 and 08 (11's export
# died on a stale CUDA error after CuMesh ran out of memory; fixed in the worker since).
set -euo pipefail

echo "== Pixal3D's weights (CPU; already there, so only the marker is checked)"
modal run ops/exp_pixal3d.py::download
echo "== Single picture, Pixal3D's single-view weights: the other eight controls"
modal run ops/exp_pixal3d.py::experiment --plan controls --out ops-out/private
echo "== Multi-view weights: 11 books again, the car's cube-fitted synthetic views, four views, the picture as main"
modal run ops/exp_pixal3d.py::experiment --plan mvchoice --out ops-out/private
du -sh ops-out/private/* || true
