#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-pixal3d branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, PIXAL3D line: its own ephemeral app (orainge-exp-pixal3d) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
#
# Run 4a: production's recipe (TRELLIS.2 '512' preview -> the picture's camera -> Pixal3D's multi-view
# weights on the picture alone -> levelled -> full export) on two backs and the three controls the
# single-view weights made worse (04 08 10 13 20). The rest of the twenty follow once these are judged.
set -euo pipefail

echo "== Pixal3D's weights (CPU; already there, so only the marker is checked)"
modal run ops/exp_pixal3d.py::download
echo "== final20 on 04 08 10 13 20"
modal run ops/exp_pixal3d.py::experiment --plan final20 --only 04,08,10,13,20 --out ops-out/private
du -sh ops-out/private/* || true
