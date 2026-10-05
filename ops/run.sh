#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paintprod: Phase 8's painter the way production would run it, in a staging app (orainge-p8-prodtest):
# modal_app.py's Painter class, called by the Trellis2 container in the middle of a final's export, through
# `make --final --paint` (the request the server sends with AI_PAINT=1). Nothing is deployed; production's
# orainge-ai keeps running what it runs.
#
# Fourth run (p8prod4-*): the painter after the code review's fixes (each final's painter waits until 150 s before
# the job's limit at most and is then cancelled; a long prompt is cut to 200 characters for the painter instead of
# failing the final; a run that keeps no views is not reported as painted). The founder's BMW prompt, then a
# 240-character sneaker prompt (the server allows 500), one after another.
#
# First the painter's weights are checked into place (they are in the orainge-models volume since the
# experiments; this writes the marker the Painter class looks for), then each prompt goes from words to a
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
run "make a bmw car m3 model blue" p8prod4-bmw || status=$?
run "a red Nike Air Jordan 1 high top sneaker in the classic Chicago colourway, white leather side panels with a red swoosh, black laces, a red collar and heel, a white midsole and a red outsole, shown as a single shoe for a product listing page" p8prod4-long || status=$?
for name in p8prod4-bmw p8prod4-long; do
  echo "== $name"
  grep -aE "\[orainge\]|\[forge3d\] paint|\[painter\]|Done:|Traceback|Error|error" "ops-out/private/$name.log" | cut -c1-300 | tail -20 || true
  if [ -d "orainge-outputs/$name" ]; then
    cp -r "orainge-outputs/$name" ops-out/private/
    python3 -c "import json,sys; s=json.load(open(sys.argv[1])); f=s['steps'].get('final',{}); print(json.dumps({'final': {k: f.get(k) for k in ('status','error','paint','projection','glass','timings','gpu_seconds','bytes')}}))" "orainge-outputs/$name/progress.json" || true
  fi
done
exit $status
