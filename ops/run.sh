#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-p8paint branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log.
set -euo pipefail

# Phase 8, run 3: all six objects (shapes kept from runs 1-2), painted with klein's single-image edit, with and
# without the view-aware prompt; the novelty check on (staging app; nothing is deployed)
ORAINGE_APP_NAME=orainge-p8-paint modal run ops/exp_paint.py::check \
  --only bmw,04,06,08,12,13 --variants edit,edit-view
