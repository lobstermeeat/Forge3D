#!/usr/bin/env bash
# What the "AI ops (Modal)" workflow runs when this is pushed (see .github/workflows/ai-ops.yml).
# Files written to ops-out/ are saved with the log, and ops-out/private/ only encrypted (ops/seal.sh).
#
# ai-ops-texcheck: checks Phase 7's texture options on the deployed job API (production, deployed
# 2 Oct 2026): a final, then a textures job for it, through the same HTTP routes the server uses.
# It deploys nothing; ops/check_textures.py runs as its own short-lived app.
set -euo pipefail
mkdir -p ops-out/private
modal run ops/check_textures.py --url https://lobstermeeat--orainge-ai-api.modal.run 2>&1 |
  grep -avE "it/s\]|s/it\]" | tail -60
