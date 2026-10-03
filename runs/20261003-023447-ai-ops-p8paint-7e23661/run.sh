#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-p8paint branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log.
set -euo pipefail

# Phase 8: the multi-view painter on the BMW, end to end (staging app; nothing is deployed)
ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check --only bmw
