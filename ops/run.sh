#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-shape: does TRELLIS.2 make cleaner shapes with more than production's finals ask of it? On the BMW, the
# Huracán, the Jordan 1 and the Coca-Cola bottle (the painter's pictures and seeds), in a staging app
# (orainge-p9-shape): production's 1024_cascade exported at 1,000,000 faces and a 4096 texture, and 1536_cascade
# with that export, beside today's finals (100,000 faces, 2048). Nothing is deployed.
#
# Second run: production's preset on the same H100 (an H100 samples differently from production's L40S), a 4096
# texture at 100,000 and 300,000 faces, and the 1,000,000-face export again (same seed, same GPU: it should repeat),
# on the BMW, the Huracán and the Jordan 1, to tell the export's part from the sampling's.
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p9-shape
status=0
modal run ops/exp_shape.py::check --only bmw,p5,p2 --variants prod,t4k,f300,faces --out ops-out/private/shape \
  >ops-out/private/modal-shape.log 2>&1 || status=$?
grep -aE "^\[shape\]|Traceback|Error|error" ops-out/private/modal-shape.log | grep -avE "it/s\]|s/it\]" |
  cut -c1-400 | tail -40 || true
du -sh ops-out/private/shape 2>/dev/null || true
exit $status
