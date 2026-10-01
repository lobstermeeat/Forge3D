#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to the ai-ops-mv branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# Phase 6, multiview line: its own ephemeral app (orainge-exp-mv) through `modal run`, never
# `modal deploy workers/modal_app.py`, which would replace the production app orainge-ai.
set -euo pipefail

echo "== Second round of variants (all 20 canonical sets are done)"
# back: prompts that describe the hidden side; cap1/cap2: the Phase 2 prompt with other seeds;
# zoom70: a wider orthographic frame for objects that overflow at 45/315 (car, books, cabin);
# elev20: cameras 20 degrees up, for top faces (donut, ramen); s30: 30 steps on two controls.
modal run ops/exp_mv.py::views --only all \
  --variants "back:prompt=back,runs=04+06+08+11+16+18;cap1:prompt=caption,seed=1,runs=04+06+08;cap2:prompt=caption,seed=2,runs=04+06+08;zoom70:extent=0.7,runs=13+11+19;elev20:elevation=20,runs=09+17+04;s30:steps=30,runs=13+01"
du -sh ops-out/private/views || true
