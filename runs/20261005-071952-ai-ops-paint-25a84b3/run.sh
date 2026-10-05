#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-paint: Phase 8, the view painter (Qwen-Image-Edit-2511 painting the final's shape from all
# round, baked into its texture), in staging apps. Nothing is deployed; production's orainge-ai keeps
# running what it runs.
#
# This run: the painter's weights into the orainge-models volume (ops/p8_download.py).
set -euo pipefail
mkdir -p ops-out/private
export ORAINGE_APP_NAME=orainge-p8-download
modal run ops/p8_download.py 2>&1 | grep -avE "it/s\]|s/it\]|MB/s\]" | tail -80
