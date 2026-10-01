#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-pixal3d branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, PIXAL3D line: its own ephemeral app (orainge-exp-pixal3d) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
set -euo pipefail

echo "== Weights already in the volume"
modal volume ls orainge-models / || true
echo "== Pixal3D's weights (CPU)"
modal run ops/exp_pixal3d.py::download
modal volume ls orainge-models /pixal3d || true
echo "== Single picture, Pixal3D's single-view weights: the four back failures and two controls"
modal run ops/exp_pixal3d.py::experiment --plan single --out ops-out/private
echo "== Multi-view weights on synthetic views of Phase 5 finals (the cameras' check)"
modal run ops/exp_pixal3d.py::experiment --plan synthetic --out ops-out/private
echo "== Multi-view weights on the MV line's six views of the four back failures"
modal run ops/exp_pixal3d.py::experiment --plan mvadapter --out ops-out/private
du -sh ops-out/private/* || true
