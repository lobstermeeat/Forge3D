#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to an ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# ai-ops-trellis-mv: Phase 6, TRELLIS.2 with extra views (ops/exp_tmv.py, app orainge-exp-tmv).
# Never `modal deploy workers/modal_app.py` here: that replaces the production app orainge-ai.
set -euo pipefail

echo "== Four views, multidiffusion, clipped views left out: the car again, and controls 19 and 05"
modal run ops/exp_tmv.py --plan guard --out ops-out/private
du -sh ops-out/private
