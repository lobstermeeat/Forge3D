#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-mv branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, multiview line: its own ephemeral app (orainge-exp-mv) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
set -euo pipefail

echo "== The other sixteen Phase 2 pictures, default settings (04, 06, 08 and 11 are done)"
modal run ops/exp_mv.py::views --only 01,02,03,05,07,09,10,12,13,14,15,16,17,18,19,20
echo "== Simple input changes on the four back failures"
modal run ops/exp_mv.py::views --only 04,06,08,11 \
  --variants "caption:prompt=caption;seed1:seed=1;g5:guidance=5;fill80:fill=0.8;s30:steps=30"
du -sh ops-out/private/views || true
