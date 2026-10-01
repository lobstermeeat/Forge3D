#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-pixal3d branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, PIXAL3D line: its own ephemeral app (orainge-exp-pixal3d) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
#
# Run 4c: the same production recipe on the single-view weights for the two flat objects the multi-view
# weights broke (06 shield: a hollow tray; 12 skateboard: a doubled deck), for the per-object comparison.
set -euo pipefail

echo "== Pixal3D's weights (CPU; already there, so only the marker is checked)"
modal run ops/exp_pixal3d.py::download
echo "== final20single on 06 12"
modal run ops/exp_pixal3d.py::experiment --plan final20single --only 06,12 --out ops-out/private
du -sh ops-out/private/* || true
