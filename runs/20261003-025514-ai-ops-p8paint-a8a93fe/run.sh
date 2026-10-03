#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-p8paint branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log.
set -euo pipefail

# Phase 8, run 2: the shapes of all six objects kept; the BMW and the arcade machine painted with five
# editing strategies (staging app; nothing is deployed)
ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check \
  --only bmw,04,06,08,12,13 --paint bmw,04 --variants edit,edit-ref,i2i-ref-91,i2i-ref-80,i2i-80
