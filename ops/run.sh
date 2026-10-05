#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paintprod: Phase 8's painter the way production would run it, in a staging app (orainge-p8-prodtest):
# modal_app.py's Painter class, called by the Trellis2 container in the middle of a final's export, through
# `make --final --paint` (the request the server sends with AI_PAINT=1). Nothing is deployed; production's
# orainge-ai keeps running what it runs.
#
# First the painter's weights are checked into place (they are in the orainge-models volume since the
# experiments; this writes the marker the Painter class looks for), then three prompts go from words to a
# painted final.
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p8-prodtest
status=0
modal run workers/modal_app.py::download_models --which painter >ops-out/private/download.log 2>&1 || status=$?
tail -5 ops-out/private/download.log || true
run() {
  modal run workers/modal_app.py::make --prompt "$1" --final --paint --run "$2" >"ops-out/private/$2.log" 2>&1
}
run "make a bmw car m3 model blue" p8prod-bmw & a=$!
run "a red Nike Air Jordan 1 sneaker" p8prod-jordan & b=$!
run "a classic Coca-Cola glass bottle" p8prod-coke & c=$!
for pid in $a $b $c; do wait "$pid" || status=$?; done
for name in p8prod-bmw p8prod-jordan p8prod-coke; do
  echo "== $name"
  grep -aE "\[orainge\]|\[forge3d\] paint|\[painter\]|Done:|Traceback|Error|error" "ops-out/private/$name.log" | cut -c1-300 | tail -20 || true
  if [ -d "orainge-outputs/$name" ]; then
    cp -r "orainge-outputs/$name" ops-out/private/
    python3 -c "import json,sys; s=json.load(open(sys.argv[1])); f=s['steps'].get('final',{}); print(json.dumps({'final': {k: f.get(k) for k in ('status','paint','projection','timings','gpu_seconds','bytes')}}))" "orainge-outputs/$name/progress.json" || true
  fi
done
exit $status
