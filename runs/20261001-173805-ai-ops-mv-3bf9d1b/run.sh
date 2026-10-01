#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-mv branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, multiview line: its own ephemeral app (orainge-exp-mv) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
set -euo pipefail

echo "== All twenty with the worker's chosen defaults (30 steps) and the Phase 2 prompt as the caption;"
echo "== plus hidden-side hints (BACK_PROMPTS) on the three they helped"
modal run ops/exp_mv.py::views --only all \
  --variants "caption30:prompt=caption;back30:prompt=back,runs=04+06+18"
du -sh ops-out/private/views || true
