#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-pixal3d branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, PIXAL3D line: its own ephemeral app (orainge-exp-pixal3d) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
#
# Run 3: the tilt. Pixal3D builds in the picture's camera frame, so an object pictured from above leans
# by that elevation. Experiment A: the multi-view weights with the picture's real camera as an absolute
# pose ("posed", 09 10 17 04 13), beside the same weights levelled afterwards ("mvlevel", 09 17).
# Experiment B (the single-view weights levelled afterwards) was judged locally on run 2's models.
set -euo pipefail

echo "== Pixal3D's weights (CPU; already there, so only the marker is checked)"
modal run ops/exp_pixal3d.py::download
echo "== Experiment A: the picture posed at its real elevation (multi-view weights), and mvlevel"
modal run ops/exp_pixal3d.py::experiment --plan posed --out ops-out/private
du -sh ops-out/private/* || true
