#!/usr/bin/env bash
# For ops/run.sh: renders the review gallery of the runs in orainge-outputs/ (where make and
# make_set copy them), with each final GLB for its 3D viewer, into ops-out/private/gallery/,
# which the workflow encrypts before saving (see ops/seal.sh).
#
#   bash ops/gallery.sh "Starter set"
set -euo pipefail

title=${1:-Orainge test gallery}
echo "== Gallery: installing three.js and Chromium"
npx --yes "$(node -p 'require("./package.json").packageManager')" install --frozen-lockfile --reporter=silent
python -m pip install --quiet "playwright>=1.47,<2" "pillow>=10"
python -m playwright install --with-deps chromium >/dev/null

echo "== Gallery: rendering"
python workers/gallery/make_gallery.py orainge-outputs -o ops-out/private/gallery --title "$title"
du -sh ops-out/private/gallery
