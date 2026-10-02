#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-pixal3d branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, PIXAL3D line: its own ephemeral app (orainge-exp-pixal3d) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
#
# Run 5: the production recipe with the worker's rule for thin, flat objects (the single-view weights
# when TRELLIS.2's preview is flat). 05 pistol (thin ratio 0.15) is the one thin object never made with
# the single-view weights; it goes first, so the swap to those weights runs on the GPU, and 01 fox (a
# control, 0.94) after it, so the swap back does too (its model should match phase2f-01). If the rule's
# runtime fails to load or build, the single-view runtime makes 05 the way 06 and 12 were made (run 4c).
set -euo pipefail

echo "== Pixal3D's weights (CPU; already there, so only the marker is checked)"
modal run ops/exp_pixal3d.py::download
echo "== final20auto (the thin rule) on 05, then 01"
if ! modal run ops/exp_pixal3d.py::experiment --plan final20auto --only 05,01 --out ops-out/private; then
  echo "== final20auto made nothing; final20single on 05 instead"
  modal run ops/exp_pixal3d.py::experiment --plan final20single --only 05 --out ops-out/private
fi
du -sh ops-out/private/* || true
