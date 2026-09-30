#!/usr/bin/env bash
# For ops/run.sh: renders the review gallery of a test set that make_set copied here, with each
# final GLB for its 3D viewer, into ops-out/private/<set>/, which the workflow encrypts before
# saving (see ops/seal.sh).
#
#   bash ops/gallery.sh orainge-outputs/starter "Orainge Starter Set" ["subtitle"]
set -euo pipefail

runs=$1
title=${2:-Orainge test gallery}
subtitle=(${3:+--subtitle "$3"})
if [ ! -d node_modules/three ] && [ ! -d apps/client/node_modules/three ]; then
  echo "== Gallery: installing three.js and Chromium"
  npx --yes "$(node -p 'require("./package.json").packageManager')" install --frozen-lockfile --reporter=silent
  python -m pip install --quiet "playwright>=1.47,<2" "pillow>=10"
  python -m playwright install --with-deps chromium >/dev/null
fi

echo "== Gallery: $title"
out="ops-out/private/$(basename "$runs")"
python workers/gallery/make_gallery.py "$runs" -o "$out" --title "$title" "${subtitle[@]}"
du -sh "$out"
