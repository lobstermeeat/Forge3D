#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed to an ai-ops branch
# (see .github/workflows/ai-ops.yml). Files written to ops-out/ are saved with the log, and
# ops-out/private/ only encrypted (see ops/seal.sh).
#
# ai-ops-trellis-mv: Phase 6, TRELLIS.2 with extra views (ops/exp_tmv.py, app orainge-exp-tmv).
# Never `modal deploy workers/modal_app.py` here: that replaces the production app orainge-ai.
set -euo pipefail

echo "== Synthetic views, smoke test: the shield (single, stochastic, multidiffusion, a preview)"
modal run ops/exp_tmv.py --plan synthetic --only 06 --out ops-out/private
python3 -c "
import json; s = json.load(open('ops-out/tmv-synthetic.json'))
bad = [n for o in s['objects'].values() for n, r in o.get('runs', {}).items() if r.get('error')] + [k for k, o in s['objects'].items() if 'error' in o]
print('errors:', bad); raise SystemExit(1 if bad else 0)
"
mv ops-out/tmv-synthetic.json ops-out/tmv-synthetic-06.json

echo
echo "== Synthetic views: the arcade machine, the camera and the car"
modal run ops/exp_tmv.py --plan synthetic --only 04,08,13 --out ops-out/private
du -sh ops-out/private
