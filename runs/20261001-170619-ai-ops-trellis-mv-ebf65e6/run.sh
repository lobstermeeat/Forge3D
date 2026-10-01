#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to an ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# ai-ops-trellis-mv: Phase 6, TRELLIS.2 with extra views (ops/exp_tmv.py, app orainge-exp-tmv).
# Never `modal deploy workers/modal_app.py` here: that replaces the production app orainge-ai.
set -euo pipefail

echo "== Four MV-Adapter views, multidiffusion: picture weight 1 / 2 / 4 and without the drawn front (06, 04, 11 at 512)"
modal volume ls orainge-outputs phase6/views 2>&1 | head -30
modal run ops/exp_tmv.py --plan real2 --out ops-out/private
echo "== Controls 01, 09, 13: single, four views multidiffusion, four views stochastic"
modal run ops/exp_tmv.py --plan controls --out ops-out/private
du -sh ops-out/private
